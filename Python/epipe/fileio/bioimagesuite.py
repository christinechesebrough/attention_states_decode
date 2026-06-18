import numpy as np
from ..preproc_ieeg.channels import find_neighbours

def read_mgrid(mgridFname, is_ielvis=True, as_dict=False, pairs_string=False):
    """

    Parameters
    ----------
    mgridFname - Name of mgrid file
    is_ielvis - If the mgrid file is formatted for iELVis
    as_dict - If True, return contacts as a dictionary rather than a list
    pairs_string - If True, contact pairs are returned as strings with dashes "-" between pairs.
                    Otherwise pairs are returned as tuples

    Returns
    -------
        contacts - list of dictionaries or dictionary of contacts and their properties
        contactPairs - List of string or tuples to represent neighboring contacts

    """

    # The text to indicate the start of a grid
    gridStartTextTemplate = '# Electrode Grid %d'

    # Open mgrid file and read contents
    with open(mgridFname, 'r') as f:
        mgridTxt = f.read()
        mgridData = mgridTxt.splitlines()

    # Get the number of electrodes expected
    nGrids = int(mgridData[6])

    # This is the data that will be returned
    grids = []
    contactPairs = []
    if as_dict:
        contacts = {}
    else:
        contacts = []

    # Seek each electrode
    for gridii in range(nGrids):

        # Find the start of each grid
        gridStartText = gridStartTextTemplate % gridii
        gridStartLine = mgridData.index(gridStartText)

        # Get basic info
        gridInfo = {
            "name": mgridData[gridStartLine + 4],
            "dimensions": [int(n) for n in mgridData[gridStartLine + 6].lstrip().split(' ')],
            "spacing": [float(n) for n in mgridData[gridStartLine + 8].lstrip().split(' ')],
            "type": int(mgridData[gridStartLine + 10]),
            "color": [float(n) for n in mgridData[gridStartLine + 16].lstrip().split(' ')]
        }
        grids.append(gridInfo)

        # If importing for ielvis then break down the electrode name
        if is_ielvis:
            gridParts = gridInfo['name'].split('_')
            gridInfo['name'] = gridParts[1]
            gridInfo['spec'] = gridParts[0][1]
            gridInfo['hem'] = gridParts[0][0]

        # Number of contacts on this grid
        dimensions = gridInfo['dimensions']
        nContacts = dimensions[0] * dimensions[1]
        gridContacts = []

        # Get info on each individual contact in this grid
        nextContactStartLine = gridStartLine + 18
        for contactii in range(nContacts):

            locInGrid = [int(n) for n in mgridData[nextContactStartLine].replace('# Electrode ', '').split(' ')]

            contactInfo = {
                "grid_row": locInGrid[0],
                "grid_column": locInGrid[1],
                "position": [float(n) for n in mgridData[nextContactStartLine + 4].lstrip().split(' ')],
                "motor": int(mgridData[nextContactStartLine + 8]),
                "sensory": int(mgridData[nextContactStartLine + 10]),
                "visual": int(mgridData[nextContactStartLine + 12]),
                "language": int(mgridData[nextContactStartLine + 14]),
                "auditory": int(mgridData[nextContactStartLine + 16]),
                "motor": int(mgridData[nextContactStartLine + 18]),
                "seizure": int(mgridData[nextContactStartLine + 22]),
                "spikes": int(mgridData[nextContactStartLine + 24]),
                "present": int(mgridData[nextContactStartLine + 26])
            }

            # Determine this contact's name
            numInGrid = dimensions[0] * contactInfo["grid_row"] + contactInfo["grid_column"] + 1
            contactName = gridInfo["name"] + str(numInGrid)

            # Add general grid info to this dictionary
            gridInfoTmp = gridInfo.copy()
            gridName = gridInfoTmp.pop('name')
            contactInfo['grid'] = gridName
            gridInfoTmp.pop('dimensions')
            gridInfoTmp.pop('spacing')

            if as_dict:
                contacts[contactName] = contactInfo
            else:
                contactInfo["name"] = contactName
                contacts.append(contactInfo)

        # Get the neighbors for this grid
        grid_layout = (np.arange(0, nContacts) + 1).reshape(dimensions[0], dimensions[1])
        my_neighbors = find_neighbours(grid_layout)
        pairNameTemplate = '{stem}{n1}-{stem}{n2}'
        for contactDict in my_neighbors:
            val = contactDict['value']
            neighbors = contactDict['neighbors']
            for neighborii in neighbors:

                # Add pair as a tuple or a string separated by a dash
                if pairs_string:
                    contact1 = min([neighborii, val])
                    contact2 = max([neighborii, val])
                    pair = pairNameTemplate.format(stem=gridName, n1=contact1, n2=contact2)
                else:
                    contact1 = gridName + str(min([neighborii, val]))
                    contact2 = gridName + str(max([neighborii, val]))
                    pair = (contact1, contact2)

                contactPairs.append(pair)


        # Append info and go to next grid
        nextContactStartLine += 37
        gridContacts.append(contactInfo)
        contacts.append(contactInfo)

    # Remove duplicates and sort
    contactPairs = list(set(contactPairs))
    contactPairs.sort()

    return contacts, contactPairs