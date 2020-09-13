# Script to call each task file sequentially
exec(open("clean.py").read())
exec(open("augment.py").read())
exec(open("classify.py").read())
