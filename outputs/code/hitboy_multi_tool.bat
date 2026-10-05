@echo off
setlocal EnableExtensions EnableDelayedExpansion
title HitBoy Multi-Tool
color 0B

:menu
cls
echo ========================================================
echo   HitBoy Multi-Tool
echo   Windows batch multi-tool suite
echo ========================================================
echo  1. Calculator
echo  2. Create folder
echo  3. List folder files
echo  4. Ping host
echo  5. System info
echo  6. Open URL
echo  7. Copy file
echo  8. Delete file
echo  9. Count lines in text file
echo 10. IP config
echo 11. Flush DNS
echo 12. Open Notepad

echo  0. Exit
echo ========================================================
set /p choice=Select tool: 

if "%choice%"=="1" goto calc
if "%choice%"=="2" goto mkdir
if "%choice%"=="3" goto list
if "%choice%"=="4" goto ping
if "%choice%"=="5" goto sysinfo
if "%choice%"=="6" goto openurl
if "%choice%"=="7" goto copyfile
if "%choice%"=="8" goto delfile
if "%choice%"=="9" goto wordcount
if "%choice%"=="10" goto ipconfig
if "%choice%"=="11" goto flushdns
if "%choice%"=="12" goto notepad

if "%choice%"=="0" goto end
echo Unknown option.
pause
goto menu

:calc
set /p expr=Expression (example 12+8*3): 
set /a result=%expr%
echo Result: %result%
pause
goto menu

:mkdir
set /p folder=Folder path to create: 
if "%folder%"=="" goto menu
mkdir "%folder%" 2>nul
if exist "%folder%" (echo Created: %folder%) else (echo Failed to create folder.)
pause
goto menu

:list
set /p folder=Folder path to list: 
if "%folder%"=="" goto menu
if not exist "%folder%" (
  echo Folder not found.
  pause
  goto menu
)
dir /b "%folder%"
pause
goto menu

:ping
set /p host=Host or IP: 
if "%host%"=="" goto menu
ping -n 4 %host%
pause
goto menu

:sysinfo
echo Computer: %COMPUTERNAME%
echo User: %USERNAME%
echo OS: %OS%
ver
echo.
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
pause
goto menu

:openurl
set /p url=URL: 
if "%url%"=="" goto menu
start "" "%url%"
echo Opened %url%
pause
goto menu

:copyfile
set /p src=Source file: 
set /p dst=Destination path: 
if "%src%"=="" goto menu
if "%dst%"=="" goto menu
copy /Y "%src%" "%dst%"
pause
goto menu

:delfile
set /p target=File to delete: 
if "%target%"=="" goto menu
if not exist "%target%" (
  echo File not found.
  pause
  goto menu
)
del /F /Q "%target%"
echo Deleted.
pause
goto menu

:wordcount
set /p target=Text file path: 
if "%target%"=="" goto menu
if not exist "%target%" (
  echo File not found.
  pause
  goto menu
)
for /f %%A in ('type "%target%" ^| find /c /v ""') do set lines=%%A
echo Lines: %lines%
pause
goto menu

:ipconfig
ipconfig /all
pause
goto menu

:flushdns
ipconfig /flushdns
pause
goto menu

:notepad
set /p target=File to edit (blank = new): 
if "%target%"=="" (
  start notepad
) else (
  start notepad "%target%"
)
goto menu


:end
echo Bye from HitBoy Multi-Tool.
endlocal
exit /b 0
