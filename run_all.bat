@echo off
REM ============================================================
REM  run_all.bat - full edge-inference thesis pipeline (PC/dev)
REM  Double-click, or run from a terminal in this folder.
REM  Uses Python 3.12 via the Windows launcher (py -3.12).
REM ============================================================
setlocal
cd /d "%~dp0"
set PY=py -3.12

echo.
echo === [0/7] installing dependencies (first run only) ===
%PY% -m pip install --quiet --disable-pip-version-check numpy pandas scikit-learn matplotlib scipy
if errorlevel 1 goto :fail

echo.
echo === [1/7] config space ===
%PY% config.py
if errorlevel 1 goto :fail

echo.
echo === [2/7] measurement sweep (synthetic power on PC) -> data\waveforms.csv ===
%PY% run_sweep.py --smoke --modes inference train
if errorlevel 1 goto :fail

echo.
echo === [3/7] build training dataset -> data\synthetic_waveforms.csv ===
%PY% -m predictor.synthetic
if errorlevel 1 goto :fail

echo.
echo === [4/7] TRAIN predictor (RQ2) -> results\predictor.pkl ===
%PY% -m predictor.train_predictor
if errorlevel 1 goto :fail

echo.
echo === [5/7] optimizer + 2D Pareto (RQ3) -> results\pareto.png ===
%PY% -m optimizer.optimize --peak-w 18 --energy-j 1.0
if errorlevel 1 goto :fail

echo.
echo === [6/7] thesis figures + 3D surface + facility scale ===
%PY% make_figures.py
%PY% pareto_3d.py --no-show
%PY% -m datacenter.scale_up --gpus 4096
%PY% export_for_matlab.py
if errorlevel 1 goto :fail

echo.
echo === [7/7] DONE ===
echo   Thesis figures : results\figures\  (fig1..fig4)
echo   2D / 3D Pareto : results\pareto.png , results\pareto_3d.png
echo   Facility scale : results\facility_power.png
echo   MATLAB surface : open  pareto_surface.m  in MATLAB (reads results\surface_data.csv)
echo.
goto :done

:fail
echo.
echo *** A step failed. Scroll up for the error. ***
echo     If 'py -3.12' was not found, run:  py --list   and edit the PY= line above.
echo.

:done
pause
endlocal
