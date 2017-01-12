// Copyright 1998-2016 Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class UnrealEnginePython : ModuleRules
{

    //Swap python versions here
    private string PythonType = "Python35";
    //private string PythonType = "Python27";

    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/")); }
    }

    protected string PythonHome
    {
        get
        {
			return Path.GetFullPath(Path.Combine(ThirdPartyPath, PythonType));
		}
	}

    public UnrealEnginePython(TargetInfo Target)
    {


        PublicIncludePaths.AddRange(
            new string[] {
                "UnrealEnginePython/Public",
				// ... add public include paths required here ...
            }
            );


        PrivateIncludePaths.AddRange(
            new string[] {
                "UnrealEnginePython/Private",
                PythonHome,
				// ... add other private include paths required here ...
			}
            );


        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "Sockets",
                "Networking",
				// ... add other public dependencies that you statically link with here ...
			}
            );


        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "InputCore",
                "Slate",
                "SlateCore",
                "MovieScene",
                "LevelSequence",
                "Projects",
				// ... add private dependencies that you statically link with here ...
			}
            );


        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );


        if (UEBuildConfiguration.bBuildEditor)
        {
            PrivateDependencyModuleNames.AddRange(new string[]{
                "UnrealEd",
                "LevelEditor"
            });
        }

        if ((Target.Platform == UnrealTargetPlatform.Win64) || (Target.Platform == UnrealTargetPlatform.Win32))
        {
            System.Console.WriteLine("Using Python at: " + PythonHome);
            PublicIncludePaths.Add(PythonHome);
            PublicAdditionalLibraries.Add(Path.Combine(PythonHome, "Lib", string.Format("{0}.lib", PythonType)));
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            if (PythonType == "Python35")
            {
                string mac_python = "/Library/Frameworks/Python.framework/Versions/3.5/";
                PublicIncludePaths.Add(Path.Combine(mac_python, "include"));
                PublicAdditionalLibraries.Add(Path.Combine(mac_python, "lib", "libpython3.5m.dylib"));
                Definitions.Add(string.Format("UNREAL_ENGINE_PYTHON_ON_MAC=3"));
            }
            else if (PythonType == "Python27") {
                string mac_python = "/Library/Frameworks/Python.framework/Versions/2.7/";
                PublicIncludePaths.Add(Path.Combine(mac_python, "include"));
                PublicAdditionalLibraries.Add(Path.Combine(mac_python, "lib", "libpython2.7.dylib"));
                Definitions.Add(string.Format("UNREAL_ENGINE_PYTHON_ON_MAC=2"));
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            if (PythonType == "Python35")
            {
                PublicIncludePaths.Add("/usr/include/python3.5m");
                PublicAdditionalLibraries.Add("/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5.so");
            }
            else if (PythonType == "Python27") {
                PublicIncludePaths.Add("/usr/include/python2.7");
                PublicAdditionalLibraries.Add("/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so");
            }
            Definitions.Add(string.Format("UNREAL_ENGINE_PYTHON_ON_LINUX"));
        }

    }
}
