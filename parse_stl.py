import pydicom
import io
import struct
import numpy


class Stl(object):
    dtype = numpy.dtype([
        ('normals', numpy.float32, (3, )),
        ('v0', numpy.float32, (3, )),
        ('v1', numpy.float32, (3, )),
        ('v2', numpy.float32, (3, )),
        ('attr', 'u2', (1, )),
    ])

    def __init__(self, header, data):
        self.header = header
        self.data = data

    @classmethod
    def from_file(cls, filename, mode='rb'):
        with open(filename, mode) as fh:
            header = fh.read(80)
            size, = struct.unpack('@i', fh.read(4))
            print(size)
            data = numpy.fromfile(fh, dtype=cls.dtype, count=size)
            return Stl(header, data)

    @classmethod
    def from_obj(cls, a):
        b_a=io.BytesIO(a)
        header = b_a.read(80)
        size, = struct.unpack('@i', b_a.read(4))
        rest=b_a.read()
        data = numpy.frombuffer(rest, dtype=cls.dtype,count=size)
        return Stl(header, data)

    def to_file(self, filename, mode='wb'):
        with open(filename, mode) as fh:
            fh.write(self.header)
            fh.write(struct.pack('@i', self.data.size))
            self.data.tofile(fh)


if __name__ == '__main__':
    # Read from STL file
    ROI_path = r'C:\Users\sanfordt\Desktop\sorted_dicoms\3648187\20180929\DCAD ROI\IM-0020-10000.dcm'
    prostate_path=r'C:\Users\sanfordt\Desktop\sorted_dicoms\3648187\20180929\DCAD STL Prostate Boundary\IM-0019-10000.dcm'

    dcm_obj = pydicom.dcmread(ROI_path)
    subs = dcm_obj[0x42011020].value[0]
    array = bytearray(subs[0x42011201].value)


    stl2=Stl.from_obj(array)
    stl2.to_file(r'C:\Users\sanfordt\Desktop\stl_out\wp2.stl')