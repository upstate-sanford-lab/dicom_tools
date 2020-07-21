import os
import numpy as np
import sys
import pydicom
import shutil
from struct import unpack
import SimpleITK as sitk
from skimage import draw
import xml.etree.ElementTree as ET
import scipy

#authors @DrSHarmon, @T_Sanf


class FindSeg:

    def __init__(self):
        self.basePATH = 'path with original images sorted with DCAD ROI as series'
        self.savePATH='path to curated dataset'

    def process_all(self):
        ''' interates over all patients in directory and creates '''

        total_lesions=0
        logger=[]
        for scn in os.listdir(self.savePATH):
            roi_path=os.path.join(self.savePATH,scn,'roi')
            if os.path.exists(roi_path) and len(os.listdir(roi_path))>0:
                print('processing lesions for scan {}'.format(scn))
                total_lesions+=1

                #make directory for lesion if one does not already exist
                if not os.path.exists(os.path.join(self.savePATH,scn,'nifti','mask')):
                    os.mkdir(os.path.join(self.savePATH,scn,'nifti','mask'))

                #set up all paths for saving
                roi_fp = os.path.join(self.savePATH, scn, 'roi', os.listdir(os.path.join(self.savePATH, scn, 'roi'))[0])
                stl_fp=os.path.join(self.savePATH,scn,'roi')
                t2_fp=os.path.join(self.savePATH,scn,'dicoms','t2',os.listdir(os.path.join(self.savePATH,scn,'dicoms','t2'))[0])
                nii_fp=os.path.join(self.savePATH,scn,'nifti','mask')

                try:
                    self.convert_stl_to_nifti(roi_fp, stl_fp, t2_fp, nii_fp)

                except:
                    print("problem with scan {}".format(scn))
                    logger+=[scn]


    def transfer_roi_file(self):
        '''transfer all files between raw dataset and curated dataset'''
        for pt in os.listdir(self.basePATH):
            for dt in os.listdir(os.path.join(self.basePATH,pt)):
                if pt+'_'+dt in os.listdir(self.savePATH):
                    for series in os.listdir(os.path.join(self.basePATH, pt, dt)):
                        if series == 'DCAD ROI':
                            print("DCAD series present for patient {}".format(pt))
                            roi_fn = [file for file in os.listdir(os.path.join(self.basePATH, pt, dt, series)) if file.endswith('.dcm')][0]
                            roi_fp = os.path.join(self.basePATH, pt, dt, series, roi_fn)
                            if not os.path.exists(os.path.join(self.savePATH,pt+'_'+dt,'roi')):
                                os.mkdir(os.path.join(self.savePATH, pt + '_' + dt, 'roi'))
                            shutil.copy2(roi_fp,os.path.join(self.savePATH, pt + '_' + dt, 'roi'))


    def convert_stl_to_nifti(self, roi_fp, stl_fp, t2_fp,nii_fp):
        ''' convert things '''

        #get STL info
        self.extract_stl(roi_fp,stl_fp)

        #get all stl files in the location
        stl_files = [file for file in os.listdir(stl_fp) if file.endswith('.stl')]
        for stl_file in stl_files:
            stl_fp_l=os.path.join(stl_fp,stl_file)

            #get vertices
            num, h, p, n, v0, v1, v2 = self.BinarySTL(stl_fp_l)
            all_verts = np.concatenate((v0,v1,v2))

            #get T2 dicom info
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(t2_fp)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            #create mask from t2 image file dimensions
            img_array = sitk.GetArrayFromImage(image)
            img_array = np.swapaxes(img_array, 2, 0)
            mask_array = np.zeros(img_array.shape)

            #this is not very elegant, basically cast all individual points to their matrix index
            for vertex in all_verts:
                ind = np.round(np.asarray(image.TransformPhysicalPointToContinuousIndex(vertex.tolist())),1)
                mask_array[int(ind[0]),int(ind[1]),int(ind[2])]=1

            #the previous step make a mask that only had boundaries of the ROI
            #now we step across every slice and fill holes to make the full tumor mask
            mask_array = np.swapaxes(mask_array, 2, 0)
            array_out=self.flood_fill_hull(mask_array)[0]

            #save binary mask and T2
            mask_out = sitk.GetImageFromArray(array_out)
            mask_out.CopyInformation(image)

            #i just wrote generic names "tumor_mask" and "t2" here, you probably will want to change it to be automatically named
            sitk.WriteImage(mask_out, os.path.join(nii_fp,stl_file.split('.stl')[0]+'.nii.gz'))

        sitk.WriteImage(image, os.path.join(nii_fp, 't2.nii.gz'))


    def BinarySTL(self,fname):
        '''read binary stl file and save'''
        fp = open(fname, 'rb')
        Header = fp.read(80)
        nn = fp.read(4)
        Numtri = unpack('i', nn)[0]
        # print nn
        record_dtype = np.dtype([
            ('normals', np.float32, (3,)),
            ('Vertex1', np.float32, (3,)),
            ('Vertex2', np.float32, (3,)),
            ('Vertex3', np.float32, (3,)),
            ('atttr', '<i2', (1,))
        ])
        data = np.fromfile(fp, dtype=record_dtype, count=Numtri)
        fp.close()

        Normals = data['normals']
        V0 = data['Vertex1']
        V1 = data['Vertex2']
        V2 = data['Vertex3']

        p = np.append(V0, V1, axis=0)
        p = np.append(p, V2, axis=0)  # list(v1)
        Points = np.array(list(set(tuple(p1) for p1 in p)))

        return Numtri, Header, Points, Normals, V0, V1, V2

    def extract_stl(self,path_to_dcm,savepath):
        '''read out the stl file'''

        # get dicom object
        dcm_obj = pydicom.dcmread(path_to_dcm)

        #make dictionary of lesion information
        lesion_info=self.describe_lesions(path_to_dcm)

        # for each lesion, extract the stl data and meta-data and save
        for i in range(len(dcm_obj[0x42011020].value)):
            lesion_dict=lesion_info[str(i+1)]
            name=self.make_name(lesion_dict)
            subs = dcm_obj[0x42011020].value[i]
            array = bytearray(subs[0x42011201].value)
            f = open(os.path.join(savepath,name+'.stl'), 'w+b')
            f.write(array)


    def describe_lesions(self, path=''):
        '''find the private tag containing lesion information and parse data as XML and store dictionary
        with keys of dictionary being the lesion number
        '''
        out_dict = {}
        dcm_obj = pydicom.dcmread(path)
        subs = dcm_obj[0x30060020].value
        for roi in subs:
            roi_dict = {}
            ROI_n = str(int(roi[0x30060022].value))
            if (0x4201, 0x1407) in roi:
                root = ET.fromstring(roi[0x42011407].value)
            elif (0x4201, 0x1403) in roi:
                root = ET.fromstring(roi[0x42011403].value)
            else:
                print('dicom tag not here!')
            for child in root:
                roi_dict[child.tag] = child.text
            out_dict[ROI_n] = roi_dict
        return (out_dict)

    def make_name(self,dict={}):
        '''create label from dictionary output of describe_lesions'''

        name=dict['COMMENTS']
        name='-'.join(name.split(' '))
        for val in ['OVERALL_SCORE','T2WTZ_SCORE','DWI_SCORE','DCE_SCORE','EPE','VOL','DIM']:
            if val in dict.keys():
                name+='_'+dict[val]
            else:
                name+='_NONE'
        return name

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

    def flood_fill_hull(self,image):
        points = np.transpose(np.where(image==1))
        hull = scipy.spatial.ConvexHull(points)
        deln = scipy.spatial.Delaunay(points[hull.vertices])
        idx = np.stack(np.indices(image.shape), axis=-1)
        out_idx = np.nonzero(deln.find_simplex(idx) + 1)
        out_img = np.zeros(image.shape)
        out_img[out_idx] = 1
        return out_img, hull

if __name__=='__main__':
    c=FindSeg()
    c.process_all()
