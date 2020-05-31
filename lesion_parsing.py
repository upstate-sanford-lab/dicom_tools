import os
import numpy as np
import struct
import pydicom
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from struct import pack, unpack



class FindSeg:

    def __init__(self):
        self.basePATH = '/Volumes/G-SPEED Shuttle TB3'

    def lets_see_prostate(self):
        RTS = os.path.join(os.path.dirname(self.basePATH), 'sorted_dicoms', '004462', '20191018','DCAD STL Prostate Boundary')
        t2 = os.path.join(os.path.dirname(self.basePATH), 'sorted_dicoms', '004462', '20191018', 'T2 Ax')
        save = os.path.join(os.path.dirname(self.basePATH), 'sorted_dicoms', '004462', '20191018', 'RTS_out')
        RTS_f = os.path.join(RTS, os.listdir(RTS)[0])
        dcm_obj = pydicom.dcmread(RTS_f)

        subs = dcm_obj[0x42011020].value[0]
        print(subs)
        array = bytearray(subs[0x42011201].value)

        with open("'/Users/sanforth/Desktop/stl/prostate2.bin", "wb") as file:
            file.write(pack("<IIIII", *bytearray()))

        print(len(array))
        print(len(array[0:80]))


        stl.Base
        t_mesh = stl.mesh.Mesh.from_file('/Users/sanforth/Desktop/stl/prostate2.stl')

        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(t_mesh.vectors))
        scale = t_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # pyplot.show()


    def lets_see_tumor(self):
        '''

        :return:
        '''

        id='004462'
        RTS = os.path.join(os.path.join(self.basePATH, 'sorted_dicoms', id, '20191018','DCAD ROI'))
        save = os.path.join(os.path.join(self.basePATH, 'sorted_dicoms', id, '20191018', 'RTS_out'))

        # get dicom object
        RTS_f = os.path.join(RTS, os.listdir(RTS)[0])
        dcm_obj = pydicom.dcmread(RTS_f)

        # get dicom info and save as binary array
        subs = dcm_obj[0x42011020].value[0]
        array = bytearray(subs[0x42011201].value)
        f = open('/Users/sanforth/Desktop/stl/tumor2.stl', 'w+b')
        f.write(array)

        num,h,p,n,v0,v1,v2=self.BinarySTL('/Users/sanforth/Desktop/stl/tumor2.stl')

        VERTICE_COUNT = num
        data = np.zeros(VERTICE_COUNT, dtype=mesh.Mesh.dtype)
        your_mesh = mesh.Mesh(data, remove_empty_areas=False)

        # The mesh normals (calculated automatically)
        your_mesh.normals=n
        # The mesh vectors
        your_mesh.v0=v0; your_mesh.v1=v1; your_mesh.v2=v2

        your_mesh.save('/Users/sanforth/Desktop/stl/tumor2_new.stl')



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




    def read_stl(self, filename):
        with open(filename, 'rb') as f:
            print(f)
            Header = f.read(80)
            nn = f.read(4)
            Numtri = struct.unpack('i', nn)[0]
            record_dtype = np.dtype([
                ('Normals', np.float32, (3,)),
                ('Vertex1', np.float32, (3,)),
                ('Vertex2', np.float32, (3,)),
                ('Vertex3', np.float32, (3,)),
                ('atttr', '<i2', (1,)),
            ])
            data = np.zeros((Numtri,), dtype=record_dtype)
            for i in range(0, Numtri, 10):
                d = np.fromfile(f, dtype=record_dtype, count=10)
                data[i:i + len(d)] = d

        # normals = data['Normals']
        v1 = data['Vertex1']
        v2 = data['Vertex2']
        v3 = data['Vertex3']
        points = np.hstack(((v1[:, np.newaxis, :]), (v2[:, np.newaxis, :]), (v3[:, np.newaxis, :])))
        return points



if __name__=='__main__':
    c=FindSeg()
    c.lets_see_tumor()