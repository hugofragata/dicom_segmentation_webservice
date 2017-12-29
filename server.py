import shutil
import cherrypy
import os
import IPaaS_utils as u

class Service(object):


    @cherrypy.expose
    def upload(self):
        '''Handle non-multipart upload'''

        filename = os.path.basename(cherrypy.request.headers['x-filename'])
        destination = os.path.join('./images/', filename[:-4]+'_new.dcm')
        with open(destination, 'wb') as f:
            shutil.copyfileobj(cherrypy.request.body, f)
        print("heck yea.")

        f = './images/'+filename[:-4]+'_new.dcm'
        print('f: '+f)

        ds = u.read_dicom(f)
        nf = u.save_PIL(f, ds)
        print('nf: '+nf)

        nnf = u.image_computing_watershed(nf)
        print('nnf: '+nnf)
        seg_dcm = u.save_segmentation(f, nnf)
        #u.read_seg(seg_dcm)

    @cherrypy.expose
    def tinker(self, f, k, dist):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        #print("******")
        #print(f,k,dist)
        try:
            k = int(float(k))
            dist = int(float(dist))
        except:
            k = 30
            dist = 0.1

        nf = u.image_computing_watershed(f, k, dist)
        print(nf)
        seg_dcm = u.save_segmentation(f[:-4]+'_new.dcm', nf)

    @cherrypy.expose
    def index(self):
        return open('index.html', 'r').read()

cherrypy.quickstart(Service(), '/', 'Service.config')