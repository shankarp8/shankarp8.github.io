import subprocess
program_list=['gpt_ft_distill_ecbd.py','gpt_ft_distill_ecbd2.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
