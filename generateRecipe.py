import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')

def create_recipe():
    text = gpt2.generate(sess, return_as_list=True, length=200)[0]
    text = "\n".join(text.split("."))
    print(text)
