
def rotator(vals,ALPHA=0,BETA=0):
    """Rotate particle set around given angles.

    Parameters
    ----------
    vals : np.array
        a Nx3 array typically consisting of
        either positions or velocities.
    ALPHA : float, optional
        First angle to rotate about
    BETA : float, optional
        Second angle to rotate about

    Examples
    --------
    rotated_pos = rotator(positions, 32.3, 55.2)

    """
    import numpy as np
    if ALPHA != 0:
        c    = np.cos(ALPHA)
        s    = np.sin(ALPHA)
        Rx   = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        vals = np.dot(Rx,vals)
    if BETA != 0:
        c    = np.cos(BETA)
        s    = np.sin(BETA)
        Ry   = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        vals = np.dot(Ry,vals)
    return vals
