; Inno Setup betiği — Desen Arama Windows kurulum sihirbazı
; Depo kökünden derlenir:  ISCC packaging\installer.iss
; Kaynak: PyInstaller çıktısı  dist\DesenArama\  (onedir)

#define AppName "Desen Arama"
#define AppVersion "1.0.0"
#define AppPublisher "ahmetvural79"
#define AppExeName "DesenArama.exe"

[Setup]
AppId={{8F3A2C10-2E4B-4E7A-9C21-DESENARAMA001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\DesenArama
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2
SolidCompression=yes
OutputDir=..\dist_installer
OutputBaseFilename=DesenAramaSetup-{#AppVersion}
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Masaüstü kısayolu oluştur"; GroupDescription: "Ek kısayollar:"

[Files]
Source: "..\dist\DesenArama\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{#AppName} Kaldır"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Uygulamayı başlat"; Flags: nowait postinstall skipifsilent
