#ue.exec('tryautopip.py')

import pip
import sys
import unreal_engine as ue
import redirect_print

#now we're ready, try to auto import pytest
try:
    import pytest
except ImportError:
	ue.log('pytest import unsuccessful')
	pip.main(['install', 'pytest'])
	#consider: call("pip install pytest", shell=True)
else:
	ue.log('pytest import successful')
