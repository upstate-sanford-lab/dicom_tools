import os
import numpy as np
import pydicom
import shutil
from struct import unpack
from scipy import ndimage
import SimpleITK as sitk
import xml.etree.ElementTree as ET


class FindSeg:

    def __init__(self):
        self.basePATH = '/Volumes/G-SPEED Shuttle TB3/sorted_dicoms'
        self.savePATH='/Users/sanforth/Desktop/awesome'

    def process_all(self):
        ''' interates over all patients in directory and creates '''

        for pt in os.listdir(self.basePATH)[0:10]:
            for dt in os.listdir(os.path.join(self.basePATH,pt)):
                for series in os.listdir(os.path.join(self.basePATH,pt,dt)):

                    if series=='DCAD ROI':
                        roi_fn=[file for file in os.listdir(os.path.join(self.basePATH,pt,dt,series)) if file.endswith('.dcm')][0]
                        roi_fp=os.path.join(self.basePATH,pt,dt,series,roi_fn)

                        #set up file structure (if it does not already exist
                        if not os.path.exists(os.path.join(self.savePATH,pt+'_'+dt)):
                            os.mkdir(os.path.join(self.savePATH,pt+'_'+dt))
                            if not os.path.exists(os.path.join(self.savePATH,pt+'_'+dt,'roi')):
                                os.mkdir(os.path.join(self.savePATH,pt+'_'+dt,'roi'))
                            if not os.path.exists(os.path.join(self.savePATH,pt+'_'+dt,'nifti')):
                                os.mkdir(os.path.join(self.savePATH,pt+'_'+dt,'nifti'))
                                if not os.path.exists(os.path.join(self.savePATH,pt+'_'+dt,'nifti','mask')):
                                    os.mkdir(os.path.join(self.savePATH,pt+'_'+dt,'nifti','mask'))
                            if not os.path.exists(os.path.join(self.savePATH, pt + '_' + dt, 'dicom')):
                                os.mkdir(os.path.join(self.savePATH, pt + '_' + dt, 'dicom'))
                                if not os.path.exists(os.path.join(self.savePATH,pt+'_'+dt,'nifti','t2')):
                                    os.mkdir(os.path.join(self.savePATH,pt+'_'+dt,'nifti','t2'))


                        #copy ROI file to new location
                        shutil.copy2(roi_fp,os.path.join(self.savePATH,pt+'_'+dt,'roi'))

                        #set up all paths for saving
                        stl_fp=os.path.join(self.savePATH,pt+'_'+dt,'roi')
                        t2_fp=os.path.join(self.savePATH,pt+'_'+dt,'dicom','t2')
                        nii_fp=os.path.join(self.savePATH,pt+'_'+dt,'nifti','mask')

                        self.convert_stl_to_nifti(roi_fp, stl_fp, t2_fp, nii_fp)


    def convert_stl_to_nifti(self, roi_fp, stl_fp, t2_fp,nii_fp):
        ''' convert things '''

        #get STL info
        self.extract_stl(roi_fp,stl_fp)
        stl_files = [file for file in os.listdir(stl_fp) if file.endswith('.stl')]
        for stl_file in stl_files:
            stl_fp_l=os.path.join(stl_fp,stl_file)
            num, h, p, n, v0, v1, v2 = self.BinarySTL(stl_fp_l)
            all_verts = np.concatenate((v0,v1,v2))

            #get T2 dicom info
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(t2_fp)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            #create mask from image file
            img_array = sitk.GetArrayFromImage(image)
            img_array = np.swapaxes(img_array, 2, 0)
            mask_array = np.zeros(img_array.shape)


            #this is not very elegant, basically cast all individual points to their matrix index
            for vertex in all_verts:
                ind = np.round(np.asarray(image.TransformPhysicalPointToContinuousIndex(vertex.tolist())),1)
                if ind[2].is_integer(): #skip indices that are inbetween z slices (interpolated probably)
                    mask_array[int(ind[0]),int(ind[1]),int(ind[2])]=1

            #the previous step make a mask that only had boundaries of the ROI
            #now we step across every slice and fill holes to make the full tumor mask
            mask_array = np.swapaxes(mask_array, 2, 0)
            for j in range(0, mask_array.shape[0]):
                slice_array = mask_array[j,:,:]
                slice_array = ndimage.binary_fill_holes(slice_array)
                mask_array[j,:,:] = slice_array.astype('uint16')

            #save binary mask and T2
            mask_out = sitk.GetImageFromArray(mask_array)
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

if __name__=='__main__':
    c=FindSeg()
    c.process_all()