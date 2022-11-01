# Reinforcement Learning Project: Pong

## Requirements
To use the version 0.18 of the gym environment it is necessary to use Python 3.7 or lower.
Also, to use torch it is necessary to use the 64 bit version of Python. 
Everything else required is in the requirements.txt file.


## Configuration
When training the agent by running "train.py", do not forget to add the "--env CartPole-v0" parameters.

When evaluating the agent by running "evaluate.py", do not forget to add the "--path models/"Name of model.pt" parameters.

There may be more parameters to come...

### Getting Pong to work
The files you need are in the "extras" folder.

The library we are using under the hood, atari-py, has a bug.
It is missing a file called 'alec_c.dll'. You need to find your installation of python
and add that file into the right folder in your Python installation. On mine it was at:
"C:\Users\aronf\AppData\Local\Programs\Python\Python37\Lib\site-packages\atari_py\ale_interface"

Once this file is in the right place you need to install the ROM.
Unzip the "Roms.rar" file, and with the python environment that you use for the 
project perform the command "python -m atari_py.import_roms 'path to folder' "
where 'path to folder' is the path to the unzipped atari ROMs.