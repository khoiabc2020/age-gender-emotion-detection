"""
Create Windows installer using Inno Setup
Generates installer script and instructions
"""

import os
from pathlib import Path

def create_inno_script():
    """Create Inno Setup script for Windows installer"""
    
    script_content = '''; Smart Retail AI - Inno Setup Installer Script
; Generated installer script for Windows

#define AppName "Smart Retail AI"
#define AppVersion "1.0.0"
#define AppPublisher "Smart Retail"
#define AppURL "https://github.com/khoiabc2020/age-gender-emotion-detection"
#define AppExeName "SmartRetailAI.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
LicenseFile=
OutputDir=installer
OutputBaseFilename=SmartRetailAI_Setup
SetupIconFile=
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "vietnamese"; MessagesFile: "compiler:Languages\\Vietnamese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 0,6.1

[Files]
Source: "dist\\SmartRetailAI.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "configs\\*"; DestDir: "{app}\\configs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "models\\*"; DestDir: "{app}\\models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\{#AppName}"; Filename: "{app}\\{#AppExeName}"
Name: "{group}\\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\\{#AppName}"; Filename: "{app}\\{#AppExeName}"; Tasks: desktopicon
Name: "{userappdata}\\Microsoft\\Internet Explorer\\Quick Launch\\{#AppName}"; Filename: "{app}\\{#AppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
procedure InitializeWizard;
begin
  WizardForm.LicenseLabel.Caption := 'Smart Retail AI - Age, Gender, and Emotion Detection System';
end;
'''
    
    installer_dir = Path('installer')
    installer_dir.mkdir(exist_ok=True)
    
    with open(installer_dir / 'SmartRetailAI.iss', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("Created Inno Setup script: installer/SmartRetailAI.iss")
    print("\nTo create installer:")
    print("1. Download Inno Setup from: https://jrsoftware.org/isdl.php")
    print("2. Open installer/SmartRetailAI.iss in Inno Setup Compiler")
    print("3. Click 'Build' -> 'Compile'")
    print("4. Installer will be created in installer/Output/")

def create_nsis_script():
    """Create NSIS script as alternative"""
    
    script_content = '''; Smart Retail AI - NSIS Installer Script
; Alternative installer using NSIS

!define APPNAME "Smart Retail AI"
!define COMPANYNAME "Smart Retail"
!define DESCRIPTION "Age, Gender, and Emotion Detection System"
!define VERSION "1.0.0"
!define HELPURL "https://github.com/khoiabc2020/age-gender-emotion-detection"
!define UPDATEURL "https://github.com/khoiabc2020/age-gender-emotion-detection"
!define ABOUTURL "https://github.com/khoiabc2020/age-gender-emotion-detection"

InstallDir "$PROGRAMFILES\\${COMPANYNAME}\\${APPNAME}"
RequestExecutionLevel admin

Page directory
Page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath "$INSTDIR"
    file /r "dist\\*"
    file /r "configs"
    file /r "models"
    
    createDirectory "$SMPROGRAMS\\${COMPANYNAME}"
    createShortCut "$SMPROGRAMS\\${COMPANYNAME}\\${APPNAME}.lnk" "$INSTDIR\\SmartRetailAI.exe"
    createShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\SmartRetailAI.exe"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "DisplayName" "${COMPANYNAME} - ${APPNAME} - ${DESCRIPTION}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\""
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "QuietUninstallString" "$\\"$INSTDIR\\uninstall.exe$\" /S"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "InstallLocation" "$INSTDIR"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "DisplayVersion" "${VERSION}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "Publisher" "${COMPANYNAME}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "HelpLink" "${HELPURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "NoModify" 1
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "NoRepair" 1
sectionEnd

function un.onInit
    SetShellVarContext all
    MessageBox MB_OKCANCEL|MB_ICONQUESTION "Are you sure you want to remove ${APPNAME} and all of its components?" IDOK next
        Abort
    next:
functionEnd

section "uninstall"
    delete "$SMPROGRAMS\\${COMPANYNAME}\\${APPNAME}.lnk"
    delete "$DESKTOP\\${APPNAME}.lnk"
    rmDir "$SMPROGRAMS\\${COMPANYNAME}"
    rmDir /r "$INSTDIR"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}"
sectionEnd
'''
    
    installer_dir = Path('installer')
    installer_dir.mkdir(exist_ok=True)
    
    with open(installer_dir / 'SmartRetailAI.nsi', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("Created NSIS script: installer/SmartRetailAI.nsi")
    print("\nTo create installer:")
    print("1. Download NSIS from: https://nsis.sourceforge.io/Download")
    print("2. Right-click installer/SmartRetailAI.nsi -> 'Compile NSIS Script'")
    print("3. Installer will be created in installer/")

if __name__ == '__main__':
    print("="*60)
    print("Smart Retail AI - Installer Script Generator")
    print("="*60)
    
    os.chdir(Path(__file__).parent)
    
    # Create both installer scripts
    create_inno_script()
    create_nsis_script()
    
    print("\n" + "="*60)
    print("Installer scripts created successfully!")
    print("="*60)
