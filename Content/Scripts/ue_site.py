import sys
import unreal_engine as ue

#get the relative paths
content = ue.get_content_dir()

pbin = content + "../Plugins/UnrealEnginePython/Binaries/Win64"
psite = pbin + "/Lib/site-packages"
#unreal_engine.get_content_dir()

pbinf = ue.convert_relative_path_to_full(pbin)
psitef = ue.convert_relative_path_to_full(psite)

#pip modules - these are now handled in c++! woohoo
#sys.path.insert(0,pbinf)
#sys.path.insert(0,psitef)
#ue.log('Added py script path: ' + pbinf)
#ue.log('Added py script path: ' + psitef)

#plugin modules - TF - todo: move from project to plugin
tfscript = content + "../Plugins/tensorflow-ue4/Content/Scripts"
tfscriptf = ue.convert_relative_path_to_full(tfscript)
ue.log('Added py script path: ' + tfscriptf)
sys.path.insert(0,tfscriptf)

ue.log('sys path is: ')
ue.log(sys.path)