#A good list http://www.nu.to.infn.it/conf/#Neutrino,%20Astroparticle%20and%20Weak%20Interaction%20Conferences

import pandas as pd
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
data = pd.read_csv('conf_2016.csv',
        names = ['ConfName', 'URL', 'AbstractDeadline', 'ConfDate', 'Location', 'Notes'], header=0,
        parse_dates=['AbstractDeadline', 'ConfDate'])
print data.sort(['AbstractDeadline'])[['ConfName', 'AbstractDeadline', 'ConfDate', 'Location', 'URL']]

