#!/usr/bin/env nu

# Get CLion build configuration and run environment from .idea/workspace.xml
#
# Optionally specify the build target name manually, e.g. pocld (default: current dir name)
export def main [target?: string] {
    let target = $target | default ($env.PWD | path basename)
    print $'target ($target)'

    let build_settings =  open .idea/workspace.xml 
        | get content 
        | where attributes == { name: CMakeSettings } 
        | get content.0.0.content.attributes 
        | update GENERATION_OPTIONS {|row| 
            $row.GENERATION_OPTIONS | split row ' '
        }
        | reject ENABLED

    let run_env = open .idea/workspace.xml
        | get content 
        | where attributes == { name: RunManager, selected: $'CMake Application.($target)' } 
        | get content.0 
        | filter {|row| 
            $row.attributes.name? == $target and $row.attributes.type? == CMakeRunConfiguration 
        }
        | get content.0
        | where tag == envs
        | get content.0.attributes

    print '>>>>>>>>>>>>>>> BUILD CONFIG <<<<<<<<<<<<<<<<<'
    print ($build_settings | table -e -t light)
    print '>>>>>>>>>>>>>>> RUN ENVIRONMENT <<<<<<<<<<<<<<'
    print ($run_env | table -e -t light)
}
