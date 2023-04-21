import subprocess
program_list=['t5_distill_ecbd.py','t5_distill_ecbd2.py','t5_distill_ecbd3.py', 't5_distill_ecbd4.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
