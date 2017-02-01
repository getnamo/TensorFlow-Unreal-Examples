
#ue.exec('simplyPythonThread.py')

import unreal_engine as ue
import time
import _thread as thread

ue.log("Attempting thread start...")

def testFunction():
	ue.log("Start threaded test...")

	for x in range(1,10):
		time.sleep(1);
		ue.log("time elapsed: " + str(x))

	ue.log("Done!")

#start thread
thread.start_new_thread(testFunction, ())

