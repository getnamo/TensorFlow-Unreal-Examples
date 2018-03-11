import sys

def clean(module='tensorflow'):
	modlist = list(sys.modules)
	n = 0
	for module in modlist:
		if module in module:
			n += 1
			del sys.modules[module]
	print('cleaned ' + str(n) + ' ' + module + ' references.')

def enum():
	for module in sys.modules:
		print(module)

def delete(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass