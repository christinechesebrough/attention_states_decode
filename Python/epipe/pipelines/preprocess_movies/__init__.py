# TODO: Functions to have
#   - Create a copy of the visual_localizer.py script at specified place



def create_copy(dname):
    """Create a copy of the visual localizer template in a specified directory

    Args:
        dname: the directory to copy the template to

    """
    import os.path as op
    from shutil import copyfile
    _thisDir = op.dirname(op.abspath(__file__))
    vicloc_tmp_fname = op.join(_thisDir, 'visual_localizer.py')
    new_fname = op.join(dname, 'visual_localizer.py')

    # Check if directory exists and if filename already exists otherwise make a copy
    if op.isfile(new_fname):
        raise FileExistsError('File %s already exists' % new_fname)
    elif not op.isdir( dname ):
        raise NotADirectoryError('Directory %s does not exist' % dname )
    else:
        print('Copying visual localizer template script as %s' % new_fname)
        copyfile(vicloc_tmp_fname, new_fname)
