NUM_FREEZED_PARAMS = 0 # NOTE: Number of layers of BACKBONE is 20.

def update_num_freezed_params(num_freezed_params):
    global NUM_FREEZED_PARAMS
    NUM_FREEZED_PARAMS = num_freezed_params

    return NUM_FREEZED_PARAMS

def get_num_freezed_params():
    global NUM_FREEZED_PARAMS
    return NUM_FREEZED_PARAMS

def log(m):
    print(f'[DEBUG]: {m}')

def freeze(param):
    param.requires_grad_(False)

def unfreeze(param):
    param.requires_grad_(True)

def is_freezed(param):
    return param.requires_grad==False

def count_num_freezed_params(module, module_name):
    num_params=0
    num_freezed_params=0
    for param in module.parameters():
        num_params+=1
        if is_freezed(param):
            num_freezed_params+=1
    assert num_params==len(list(module.parameters()))
    log(f'Number of freezed params of module {module_name}: {num_freezed_params}/{num_params}')
    return num_freezed_params, num_params

def check_freezing(num_freezed_params, module):
    # NOTE: Check 1 - Using LIST.
    all_params=list(module.parameters())
    for i in range(num_freezed_params):
        assert is_freezed(all_params[i])==True
    
    for i in range(num_freezed_params,len(all_params)):
        assert is_freezed(all_params[i])==False

    # NOTE: Check 2 - Using GENERATOR.
    idx=0
    for param in module.parameters():
        if idx<num_freezed_params:
            assert is_freezed(param)==True
        else:
            assert is_freezed(param)==False
        idx+=1

def freeze_entire_module(module):
    for param in module.parameters():
        freeze(param)
        assert is_freezed(param)==True

def check_freezed_entire_module(module):
    for param in module.parameters():
        assert is_freezed(param)==True


def freeze_module(num_freezed_params,module):
    all_params=list(module.parameters())
    for param in all_params[:num_freezed_params]:
        freeze(param) 
        assert is_freezed(param)==True 

    for param in all_params[num_freezed_params:]:
        unfreeze(param) 
        assert is_freezed(param)==False