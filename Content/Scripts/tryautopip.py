#ue.exec('tryautopip.py')

import pip
import sys
import unreal_engine as ue

class Redirector(object):
    def __init__(self):
        pass

    def write(self, message):
        ue.log(message)

    def flush(self):
    	sys.stdout.flush
    def splitlines(self):
    	sys.stdout.splitlines

sys.stdout = Redirector()
#sys.stderr = Redirector()

#now we're ready, try to auto import pytest
try:
    import pytest
except ImportError:
	ue.log('pytest import unsuccessful')
	pip.main(['install', 'pytest'])
	#consider: call("pip install pytest", shell=True)
else:
	ue.log('pytest import successful')
