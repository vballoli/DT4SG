#!/bin/sh

module load python/3.9.12
source DT4SG/bin/activate

#bash Run__gl_gpu.sh Run__offline.sh dt sac 001 test1
#bash Run__gl.sh Run__offline.sh 'td3+bc' sac 001 test2

bash Run__gl_gpu.sh Run__offline.sh dt random 001 0
bash Run__gl_gpu.sh Run__offline.sh dt random 001 1
bash Run__gl_gpu.sh Run__offline.sh dt random 001 2

bash Run__gl_gpu.sh Run__offline.sh cql random 001 0
bash Run__gl_gpu.sh Run__offline.sh cql random 001 1
bash Run__gl_gpu.sh Run__offline.sh cql random 001 2

bash Run__gl.sh Run__offline.sh 'td3+bc' random 001 0
bash Run__gl.sh Run__offline.sh 'td3+bc' random 001 1
bash Run__gl.sh Run__offline.sh 'td3+bc' random 001 2

#bash Run__gl_gpu.sh Run__offline.sh dt td3 001 0
#bash Run__gl_gpu.sh Run__offline.sh dt td3 001 1
#bash Run__gl_gpu.sh Run__offline.sh dt td3 001 2
#
#bash Run__gl.sh Run__offline.sh 'td3+bc' td3 001 0
#bash Run__gl.sh Run__offline.sh 'td3+bc' td3 001 1
#bash Run__gl.sh Run__offline.sh 'td3+bc' td3 001 2
#
#bash Run__gl_gpu.sh Run__offline.sh dt discrete_random 001 0
#bash Run__gl.sh Run__offline.sh 'td3+bc' discrete_random 001 0


