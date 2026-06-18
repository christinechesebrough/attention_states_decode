import re
from itertools import compress
import numpy as np
from mne import set_bipolar_reference, set_eeg_reference

def find_neighbours(gridMatrix):
    """
    Return the neighboring elements for each value in a matrix

    Parameters
    ----------
    gridMatrix - 2D list or numpy array representing the layout

    Returns
    -------
        list of dictionaries each containing two keys
            "value": a single value in the matrix
            "neighbors": The values it neighbors in the matrix
    """
    neighbors = []
    rows, cols = len(gridMatrix), len(gridMatrix[0])

    for i in range(rows):
        for j in range(cols):
            new_neighbors = []
            row_start, row_end = max(0, i-1), min(rows, i+2)
            col_start, col_end = max(0, j-1), min(cols, j+2)
           
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    if (r == i and c != j) or (r != i and c == j):
                        new_neighbors.append(gridMatrix[r][c])

            neighbors.append({
                "value": gridMatrix[i][j],
                "neighbors": new_neighbors
            })
            
    return neighbors


def reref_avg(raw, copy=True):
    """Average reference only using good channels"""

    avg,_ = set_eeg_reference(raw, ref_channels='average', copy=copy)
    
    avg.info['bads'] = [ch.item() if isinstance(ch, np.str_) 
                        else ch 
                        for ch in avg.info['bads']]
    return avg



def reref_bipolar(raw, elecs=None, anodes=None, cathodes=None, copy=False):
    """Set bipolar reference"""

    if elecs == None:
        elecs = raw.ch_names

    raw_bad_chans = raw.info['bads']
    bip_bad_chans = []

    if anodes == None or cathodes == None:
        
        anodes = []
        cathodes = []
        elecNames = elecs
        elecArrays = {}
        for ename in elecNames:
            grp_name = re.sub(r'\d+$', '', ename)
            if grp_name not in elecArrays.keys():
                elecArrays[grp_name] = 0
            elecArrays[grp_name] += 1

        for el in elecArrays.keys():
            nElecs = elecArrays[el]
            for ii in range(nElecs):
                elec1 = el + str(ii + 1)
                elec2 = el + str(ii + 2)
                if (elec1 in elecNames) and (elec2 in elecNames):
                    anodes.append(elec1)
                    cathodes.append(elec2)
                    if (elec1 in raw_bad_chans) or (elec2 in raw_bad_chans):
                        bip_bad_chans.append(elec1 + '-' + elec2)

        bip = set_bipolar_reference(raw, anodes, cathodes, copy=copy)
        bip.info['bads'] = bip_bad_chans
    
    else:
        bip = set_bipolar_reference(raw, anodes, cathodes, copy=copy)
        
    return bip


# Break out function for creating list of biplolar pairs for custom options 
# step = 2 -> skip a contact for each pair (sometimes useful for natus recordings)
def bipolar_list(elecs, step):
    
    anodes = []
    cathodes = []
    elecNames = elecs
    elecArrays = {}
    for ename in elecNames:
        grp_name = re.sub(r'\d+$', '', ename)
        if grp_name not in elecArrays.keys():
            elecArrays[grp_name] = 0
        elecArrays[grp_name] += 1
    
    for el in elecArrays.keys():
        nElecs = elecArrays[el]
        for ii in range(nElecs):    
            if step == 1:
                elec1 = el + str(ii + 1)
                elec2 = el + str(ii + 2)
            elif step == 2:
                elec1 = el + str(2*ii + 1)
                elec2 = el + str(2*(ii+1) + 1)
            if (elec1 in elecNames) and (elec2 in elecNames):
                anodes.append(elec1)
                cathodes.append(elec2)
                
    return anodes, cathodes

# Closest white matter reference
def reref_white_matter(raw, ptd, coords, ptd_thresh=-0.8, copy=True):
    
    # Labels
    labels = raw.ch_names
    
    # White matter channels
    idx_wm = ptd < ptd_thresh
         
    coords_wm = coords[idx_wm, :]
    labels_wm = list(compress(labels, idx_wm))
    
    # Remove bad channels
    idx_bad = np.in1d(labels_wm, raw.info['bads'])
    
    coords_wm = coords_wm[np.invert(idx_bad), :]
    labels_wm = list(compress(labels_wm, np.invert(idx_bad)))
    
    coords_not_wm = coords[np.invert(np.in1d(labels, labels_wm)), :]
    labels_not_wm = list(compress(labels, np.invert(np.in1d(labels, labels_wm))))
    
    refs = [None] * len(labels_not_wm)
    
    for ch in range(len(labels_not_wm)):
        
        idx_closest = np.argmin(np.sqrt(
            np.sum((coords_wm - coords_not_wm[ch, :])**2, 
                   axis=1)))
        
        refs[ch] = labels_wm[idx_closest]
        
    # Perform rereferencing    
    cwm = set_bipolar_reference(raw, labels_not_wm, refs, copy=copy)
    
    cwm.drop_channels(list(compress(cwm.ch_names, np.isin(cwm.ch_names, labels_wm))))
        
    cwm.info['bads'] = [ch.item() if isinstance(ch, np.str_) 
                        else ch 
                        for ch in cwm.info['bads']]
    
    return cwm, refs

# Closest white matter reference
def reref_laplacian(raw, skip_bads=True, copy=True):
    
    ch_names = raw.ch_names
    bad_chans = raw.info['bads']
    
    # Get electrodes
    elecs = []  
    for ch in ch_names:
        elec = re.sub(r'\d+$', '', ch)
        if not np.isin(elec, elecs):
            elecs.append(elec)
            
    # Create a dictionarly with references for each channel
    # At the edges: bipolar, everywhere else: average of neiboring channels
    refs = {}
    bads_ref = []
    
    for e in elecs:
        
        ch_elec = list(compress(ch_names, [e in ch for ch in ch_names]))
        
        for ic,che in enumerate(ch_elec):
            
            if ic == 0:
                if skip_bads and np.sum(np.isin(ch_elec[ic+1], bad_chans)) != 0:
                    refs[che] = []
                else:
                    refs[che] = ch_elec[ic+1]
            elif ic == len(ch_elec)-1:
                if skip_bads and np.sum(np.isin(ch_elec[ic-1], bad_chans)) != 0:
                    refs[che] = []
                else:
                    refs[che] = ch_elec[ic-1]
            else:
                if skip_bads and np.sum(np.isin(ch_elec[ic-1], bad_chans)) != 0:
                    refs[che] = ch_elec[ic+1]
                if skip_bads and np.sum(np.isin(ch_elec[ic+1], bad_chans)) != 0:
                    refs[che] = ch_elec[ic-1]
                if skip_bads and np.sum(np.isin([ch_elec[ic-1], ch_elec[ic+1]], bad_chans)) != 0:
                    refs[che] = []
                else:
                    refs[che] = [ch_elec[ic-1], ch_elec[ic+1]]
            
            if np.sum(np.isin(refs[che], bad_chans)) != 0:
                bads_ref.append(che)
             
    # Set the reference
    lap,_ = set_eeg_reference(raw, ref_channels=refs, copy=copy)
    
    # Update bad channels 
    if not skip_bads:
        bads_combined = list(np.unique(np.concatenate((raw.info['bads'], lap.info['bads']))))
        bads_combined = [s.item() for s in bads_combined]
        lap.info['bads'] = bads_combined
        
    lap.info['bads'] = [ch.item() if isinstance(ch, np.str_) 
                        else ch 
                        for ch in lap.info['bads']]
     
    return lap, refs



import numpy as np
import mne
from itertools import compress

def reref_white_matter_strict(
    raw,
    ptd,
    coords,
    labels=None,
    wm_ptd_thresh=-0.8,
    gm_ptd_thresh=0.2,
    min_ref_dist_mm=0.0,
    drop_refs=True,
    copy=True,
    strict=True,
    return_meta=False,
):
    """
    White-matter (WM) bipolar rereference with *explicit* GM targeting and extra control
    (mimics the structure of the original `reref_white_matter`, but matches your inline logic).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object containing channels referenced by `labels` (defaults to raw.ch_names).
    ptd : array-like, shape (n_channels,)
        PTD values aligned to `labels` order. (Typically in [-1, 1].)
    coords : array-like, shape (n_channels, 3)
        Coordinates aligned to `labels` order, in mm (consistent with min_ref_dist_mm).
    labels : list[str] | None
        Channel labels aligned with ptd/coords. If None, uses raw.ch_names.
    wm_ptd_thresh : float
        WM selection threshold: WM if ptd <= wm_ptd_thresh.
    gm_ptd_thresh : float
        GM selection threshold: GM targets if ptd >= gm_ptd_thresh.
    min_ref_dist_mm : float
        If > 0, prefer WM refs at least this far from target; fall back to closest if none meet.
    drop_refs : bool
        Passed to mne.set_bipolar_reference. True drops the cathode channels from output.
    copy : bool
        Passed to mne.set_bipolar_reference.
    strict : bool
        If True, raise if zero GM or zero WM channels after exclusions. If False, allow empty.
    return_meta : bool
        If True, returns extra metadata (wm_labels, gm_labels, etc.)

    Returns
    -------
    bp_raw : mne.io.BaseRaw
        Raw containing ONLY the newly created bipolar channels (picked by bp_names).
    refs : list[str]
        List of chosen WM reference channel names, in the same order as gm_labels.
    (optional) meta : dict
        Extra details if return_meta=True.
    """
    # -----------------------------
    # Basic checks / alignment
    # -----------------------------
    if labels is None:
        labels = list(raw.ch_names)
    else:
        labels = list(labels)

    ptd = np.asarray(ptd, dtype=float)
    coords = np.asarray(coords, dtype=float)

    if ptd.ndim != 1:
        raise ValueError(f"`ptd` must be 1D, got shape {ptd.shape}.")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"`coords` must be shape (n, 3), got {coords.shape}.")
    if len(labels) != len(ptd) or len(labels) != coords.shape[0]:
        raise ValueError(
            f"Alignment mismatch: len(labels)={len(labels)}, len(ptd)={len(ptd)}, coords.shape[0]={coords.shape[0]}"
        )

    # -----------------------------
    # Determine bad channels (exclude from BOTH pools)
    # -----------------------------
    bads = set(raw.info.get("bads", []))
    not_bad = np.array([ch not in bads for ch in labels], dtype=bool)

    # -----------------------------
    # Select WM refs and GM targets using PTD thresholds
    # -----------------------------
    idx_wm = (ptd <= wm_ptd_thresh) & not_bad
    idx_gm = (ptd >= gm_ptd_thresh) & not_bad

    wm_labels = np.array(labels, dtype=object)[idx_wm]
    wm_coords = coords[idx_wm, :]

    gm_labels = np.array(labels, dtype=object)[idx_gm]
    gm_coords = coords[idx_gm, :]

    if strict and len(wm_labels) == 0:
        raise ValueError(
            f"No WM channels found with ptd <= {wm_ptd_thresh} (after excluding bads={len(bads)}). "
            "Consider relaxing wm_ptd_thresh (e.g., -0.5 or 0.0) or verify PTD alignment."
        )
    if strict and len(gm_labels) == 0:
        raise ValueError(
            f"No GM channels found with ptd >= {gm_ptd_thresh} (after excluding bads={len(bads)}). "
            "Consider relaxing gm_ptd_thresh (e.g., 0.0) or verify PTD alignment."
        )

    # If not strict and either pool is empty, return gracefully
    if len(wm_labels) == 0 or len(gm_labels) == 0:
        empty = raw.copy() if copy else raw
        if return_meta:
            return empty, [], dict(wm_labels=wm_labels.tolist(), gm_labels=gm_labels.tolist())
        return empty, []

    # -----------------------------
    # For each GM channel, choose nearest WM ref (with optional min distance)
    # -----------------------------
    refs = []
    for c in range(gm_coords.shape[0]):
        d = np.sqrt(np.sum((wm_coords - gm_coords[c, :]) ** 2, axis=1))

        if min_ref_dist_mm and min_ref_dist_mm > 0:
            d2 = d.copy()
            d2[d2 < float(min_ref_dist_mm)] = np.inf
            if np.all(np.isinf(d2)):
                idx_closest = int(np.argmin(d))   # fallback: absolute closest
            else:
                idx_closest = int(np.argmin(d2))  # closest that meets min distance
        else:
            idx_closest = int(np.argmin(d))

        refs.append(str(wm_labels[idx_closest]))

    # Bipolar channel names "GM-WM"
    bp_names = [f"{a}-{b}" for a, b in zip(gm_labels.tolist(), refs)]

    # -----------------------------
    # Perform rereferencing (bipolar)
    # -----------------------------
    bp = mne.set_bipolar_reference(
        raw,
        anode=list(map(str, gm_labels.tolist())),
        cathode=list(map(str, refs)),
        ch_name=bp_names,
        drop_refs=drop_refs,
        copy=copy,
    )

    # Keep only the created bipolar channels (mirrors your inline `.pick(bp_names)`)
    bp_raw = bp.copy().pick(bp_names) if copy else bp.pick(bp_names)

    # Normalize bads to pure Python strings (mirrors your original function’s cleanup)
    bp_raw.info["bads"] = [
        ch.item() if isinstance(ch, np.str_) else ch for ch in bp_raw.info.get("bads", [])
    ]

    if return_meta:
        meta = dict(
            wm_ptd_thresh=wm_ptd_thresh,
            gm_ptd_thresh=gm_ptd_thresh,
            min_ref_dist_mm=min_ref_dist_mm,
            n_wm=int(len(wm_labels)),
            n_gm=int(len(gm_labels)),
            bads=list(bads),
            wm_labels=wm_labels.tolist(),
            gm_labels=gm_labels.tolist(),
            bp_names=bp_names,
        )
        return bp_raw, refs, meta

    return bp_raw, refs