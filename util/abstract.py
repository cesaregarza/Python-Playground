from abc import ABCMeta

class DummyAttribute:
    pass

def abstractattribute(obj=None):
    if obj is None:
        obj = DummyAttribute()
    obj.__is_abstract_attribute__ = True
    return obj

class ABCPlus(ABCMeta):

    def __call__(cls, *args, **kwargs):
        instance = ABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstract_attribute__', False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with abstract attributes {}".format(
                    cls.__name__,
                    ', '.join(abstract_attributes)
                )
            )
        return instance