#shortcut lib for re-running quick functions, auto-reloads modules
import importlib

class UPyRun(object):
	def __init__(self):
		super(UPyRun, self).__init__()
		self.module = None
	def r(self, moduleName):
		if not self.module:
			self.module = importlib.import_module(moduleName)
		else:
			importlib.reload(self.module)
		return self.module

#forward wrappers
_inst = UPyRun()
r = _inst.r
run = _inst.r
