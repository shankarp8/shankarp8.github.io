import subprocess
program_list=['gpt_ft_distill_multiple_ecbd.py','gpt_ft_distill_multiple_ecbd2.py','gpt_ft_distill_multiple_ecbd3.py','gpt_ft_distill_multiple_ecbd4.py','gpt_ft_distill_multiple_ecbd5.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
