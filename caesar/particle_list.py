
class ParticleList(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return getattr(instance, '_%s' % self.name)

    def __set__(self, instance, value):
        setattr(instance, '_%s' % self.name, value)

class ParticleListContainer(object):
    halo_dmlist  = ParticleList('halo_dmlist')
    halo_glist   = ParticleList('halo_glist')
    halo_slist   = ParticleList('halo_slist')

    galaxy_glist = ParticleList('galaxy_glist')
    galaxy_slist = ParticleList('galaxy_slist')

    def __init__(self, obj):
        self.obj = obj
