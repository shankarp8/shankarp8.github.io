import subprocess
program_list=['t5_curricula_ecbd.py', 't5_curricula_ecbd2.py','t5_curricula_ecbd3.py','t5_curricula_ecbd4.py','t5_curricula_ecbd5.py', 't5_curricula_ecbd6.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
