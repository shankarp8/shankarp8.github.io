import subprocess
program_list=['gpt_ft_distill_ecbd5.py','gpt_ft_distill_ecbd6.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
