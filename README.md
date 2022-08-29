# Install 
## Install VisualStudio BuildTool
以下サイトから、buildtoolのexeをダウンロードし、実行  
デスクトップ用のC++環境にチェックをつけ、インストール
https://visualstudio.microsoft.com/ja/downloads/#build-tools-for-visual-studio-2022

インストール後にコンパイラにPATHを通す  
for compiler (cl.exe)  
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\x.x.x\bin\Hostx64\x64

さらに、LIB, INCLUDE, LIBPATHも通す  
x64 native tools command prompt VS 2022 を起動  
以下を実行し、出力結果を環境変数に貼り付け  
echo %INCLUDE%  
echo %LIB%  
echo %LIBPATH%  

CMAKEにもパスを通す  
for cmake (cmake.exe)  
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin

## Install Requirements
```commandline
pip install numpy Cython cmake onnxruntime
pip install insightface
pip install opencv-contrib-python
```

# How to use
1. download insightface models and set at models
    link: https://drive.google.com/file/d/1pKIusApEfoHKDjeBTXYB3yOQ0EtTonNE/view?usp=sharing
2. run main.py
![img.png](readme/img.png)