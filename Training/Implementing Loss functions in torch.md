
# Built in Loss functions


# Custom Loss functions

Best approach is to make a class for your loss function. The class should inherit from nn.Module


### Forward #

By default, every subclass of nn.Module is required to provide an implementation for forward().
However, the forward() function in the loss class is different to the forward function in the model class.