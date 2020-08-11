
def get_seed_name(threshhold, min_len):
    name  =('t_'   + '{:01.02f}'.format(threshhold) \
             + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
    return name
