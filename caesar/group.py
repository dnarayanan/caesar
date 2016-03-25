

class Group(object):
    def __init__(self,obj):
        self.particle_indexes = []
        self.obj = obj
        
class Galaxy(Group):
    def __init__(self,obj):
        super(Galaxy, self).__init__(obj)
class Halo(Group):
    def __init__(self,obj):
        super(Halo, self).__init__(obj)

def create_new_group(obj, group_type):
    if group_type == 'halo':
        return Halo(obj)
    elif group_type == 'galaxy':
        return Galaxy(obj)
