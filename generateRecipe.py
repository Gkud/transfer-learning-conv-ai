import gpt_2_simple as gpt2


def create_recipe():
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='run1')

    text = gpt2.generate(sess)
    text = "\n".join(text.split("."))
    print(text)

