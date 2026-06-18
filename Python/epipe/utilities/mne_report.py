def create_mne_report_table(vals,report=None,name='custom_table',tags=[]):
    """Create a custom html table to add to a MNE report

    Parameters
    ----------
    vals : dict | OrderedDict
        key-value pairs of what to add to the table
    report : Report | None
        The mne Report object to add the table to. If None then will return an html string
    name : str
        Name to give the table
    tags : list
        Tags to give the table

    Returns
    -------
    str | Report

    """
    import mne

    # Strings for tables in mne Reports
    html_table_start = """
    <table class="table table-hover table-striped table-sm table-responsive small"><tbody>
    """

    html_table_end = """
    </tbody></table>
    """

    html_table_row = """
    <tr>
    <th>{}</th>
    <td>{}</td>
    </tr>
    """

    full_html_string = html_table_start

    for k in vals.keys():
        new_row = html_table_row.format(k,vals[k])
        full_html_string += new_row

    full_html_string += html_table_end

    if isinstance(report,mne.report.report.Report):
        report.add_html(full_html_string,name,tags=tags)
        return report
    else:
        return full_html_string