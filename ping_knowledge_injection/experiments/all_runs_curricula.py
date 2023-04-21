import subprocess
program_list=['t5_curricula_entity_inferences.py', 't5_curricula_entity_inferences2.py','t5_curricula_entity_inferences3.py','t5_curricula_entity_inferences4.py','t5_curricula_entity_inferences5.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
