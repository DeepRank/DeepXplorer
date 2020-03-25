import viztools
from PyQt5 import QtWidgets
from h5xplorer.menu_tools import *
from deeprank.tools import sparse
from pdb2sql import pdb2sql
from deeprank.learn import rankingMetrics
import numpy as np

def context_menu(self, treeview, position):

    """Generate a right-click menu for the items"""

    # make sure tha there is only one item selected
    all_item = get_current_item(self,treeview,single=False)

    if len(all_item) == 1:

        item = all_item[0]

        try:

            _type = self.root_item.data_file[item.name].attrs['type']

            if _type == 'molecule':
                molgrp = self.root_item.data_file[item.name]
                _context_mol(item,treeview,position,molgrp)

            if _type == 'sparse_matrix':
                _context_sparse(item,treeview,position)

            if _type == 'epoch':
                _task = self.root_item.data_file[item.name].attrs['task']
                _context_one_epoch(item,treeview,position,_task)

            if _type == 'losses':
                _context_losses(item,treeview,position)

        except Exception as inst:
            print(type(inst))
            print(inst)
            return

    else :

        _type = np.array([self.root_item.data_file[item.name].attrs['type'] for item in all_item])
        epoch_item = [item for item in all_item if self.root_item.data_file[item.name].attrs['type'] == 'epoch' ]
        haddock_item = [item for item in all_item if self.root_item.data_file[item.name].attrs['type'] == 'haddock' ]

        _context_multiple_epoch_multilevel(epoch_item,treeview,position,haddock_item)


def _context_mol(item,treeview,position,molgrp):

    menu = QtWidgets.QMenu()
    actions = {}
    list_operations = ['Load in PyMol','Load in VMD','PDB2SQL']

    for operation in list_operations:
        actions[operation] = menu.addAction(operation)
    action = menu.exec_(treeview.viewport().mapToGlobal(position))

    _,cplx_name, mol_name = item.name.split('/')
    mol_name = mol_name.replace('-','_')

    if action == actions['Load in VMD']:
        viztools.create3Ddata(mol_name, molgrp)
        viztools.launchVMD(mol_name)

    if action == actions['Load in PyMol']:
        viztools.create3Ddata(mol_name,molgrp)
        viztools.launchPyMol(mol_name)

    if action == actions['PDB2SQL']:
        db = pdb2sql(molgrp['complex'].value)
        treeview.emitDict.emit({'sql_' + item.basename: db})

def _context_sparse(item,treeview,position):

    menu = QtWidgets.QMenu()
    list_operations = ['Load Matrix','Plot Histogram']
    action,actions = get_actions(treeview,position,list_operations)

    name = item.basename + '_' + item.name.split('/')[2]

    if action == actions['Load Matrix']:

        subgrp = item.data_file[item.name]
        data_dict = {}
        if not subgrp.attrs['sparse']:
            data_dict[item.name] =  subgrp['value'].value
        else:
            molgrp = item.data_file[item.parent.parent.parent.name]
            grid = {}
            lx = len(molgrp['grid_points/x'].value)
            ly = len(molgrp['grid_points/y'].value)
            lz = len(molgrp['grid_points/z'].value)
            shape = (lx,ly,lz)
            spg = sparse.FLANgrid(sparse=True,index=subgrp['index'].value,value=subgrp['value'].value,shape=shape)
            data_dict[name] =  spg.to_dense()
        treeview.emitDict.emit(data_dict)

    if action == actions['Plot Histogram']:

        value = item.data_file[item.name]['value'].value
        data_dict = {'value':value}
        treeview.emitDict.emit(data_dict)

        cmd = "%matplotlib inline\nimport matplotlib.pyplot as plt\nplt.hist(value,25)\nplt.show()\n"
        data_dict = {'exec_cmd':cmd}
        treeview.emitDict.emit(data_dict)

def _context_one_epoch(item,treeview,position,task):

    menu = QtWidgets.QMenu()
    actions = {}

    if task == 'reg':

        list_operations = ['Scatter Plot','Hit Rate']
        action,actions = get_actions(treeview,position,list_operations)

        if action == actions['Scatter Plot']:

            values = []
            train_out = item.data_file[item.name+'/train/outputs'].value
            train_tar = item.data_file[item.name+'/train/targets'].value
            values.append([x for x in train_out])
            values.append([x for x in train_tar])


            valid_out = item.data_file[item.name+'/valid/outputs'].value
            valid_tar = item.data_file[item.name+'/valid/targets'].value
            values.append([x for x in valid_tar])
            values.append([x for x in valid_out])


            if 'test' in item.data_file[item.name]:
                test_out = item.data_file[item.name+'/test/outputs'].value
                test_tar = item.data_file[item.name+'/test/targets'].value
                values.append([x for x in test_tar])
                values.append([x for x in test_out])

            vmin = np.array([x for a in values for x in a]).min()
            vmax = np.array([x for a in values for x in a]).max()
            delta = vmax-vmin
            values.append([vmax + 0.1*delta])
            values.append([vmin - 0.1*delta])

            data_dict = {'_values':values}
            treeview.emitDict.emit(data_dict)

            data_dict = {}
            cmd  = "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
            cmd += "fig,ax = plt.subplots()\n"
            cmd += "ax.scatter(_values[0],_values[1],c='red',label='train')\n"
            cmd += "ax.scatter(_values[2],_values[3],c='blue',label='valid')\n"
            if 'test' in item.data_file[item.name]:
                cmd += "ax.scatter(_values[4],_values[5],c='green',label='test')\n"
            cmd += "legen = ax.legend(loc='upper left')\n"
            cmd += "ax.set_xlabel('Targets')\n"
            cmd += "ax.set_ylabel('Predictions')\n"
            cmd += "ax.plot([_values[-2],_values[-1]],[_values[-2],_values[-1]])\n"
            cmd += "plt.show()\n"
            data_dict['exec_cmd'] = cmd
            treeview.emitDict.emit(data_dict)

        if action == actions['Hit Rate']:

            values = []
            train_hit = item.data_file[item.name+'/train/hit'].value
            valid_hit = item.data_file[item.name+'/valid/hit'].value
            values.append(rankingMetrics.hitrate(train_hit))
            values.append(rankingMetrics.hitrate(valid_hit))
            if 'test' in item.data_file[item.name]:
                test_hit = item.data_file[item.name+'/test/hit'].value
                values.append(rankingMetrics.hitrate(test_hit))

            data_dict = {'_values':values}
            treeview.emitDict.emit(data_dict)

            data_dict = {}
            cmd  = "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
            cmd += "plt.plot(_values[0],c='red',label='train')\n"
            cmd += "plt.plot(_values[1],c='blue',label='valid')\n"
            if 'test' in item.data_file[item.name]:
                cmd += "plt.plot(_values[2],c='green',label='test')\n"
            cmd += "legen = ax.legend(loc='upper left')\n"
            cmd += "ax.set_xlabel('Top M')\n"
            cmd += "ax.set_ylabel('Hit rate')\n"
            cmd += "plt.show()\n"
            data_dict['exec_cmd'] = cmd
            treeview.emitDict.emit(data_dict)


    elif task == 'class':
        list_operations = ['Hit Rate']
        action,actions = get_actions(treeview,position,list_operations)

        if action == actions['Hit Rate']:

            values = []
            train_hit = item.data_file[item.name+'/train/hit'].value
            valid_hit = item.data_file[item.name+'/valid/hit'].value
            values.append(rankingMetrics.hitrate(train_hit))
            values.append(rankingMetrics.hitrate(valid_hit))
            if 'test' in item.data_file[item.name]:
                test_hit = item.data_file[item.name+'/test/hit'].value
                values.append(rankingMetrics.hitrate(test_hit))

            data_dict = {'_values':values}
            treeview.emitDict.emit(data_dict)

            data_dict = {}
            cmd  = "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
            cmd += "fig,ax = plt.subplots()\n"
            cmd += "plt.plot(_values[0],c='red',label='train')\n"
            cmd += "plt.plot(_values[1],c='blue',label='valid')\n"
            if 'test' in item.data_file[item.name]:
                cmd += "plt.plot(_values[2],c='green',label='test')\n"
            cmd += "legen = ax.legend(loc='upper left')\n"
            cmd += "ax.set_xlabel('Top M')\n"
            cmd += "ax.set_ylabel('Hit rate')\n"
            cmd += "plt.show()\n"
            data_dict['exec_cmd'] = cmd
            treeview.emitDict.emit(data_dict)


def _context_multiple_epoch(epoch_items,treeview,position,haddock_item=None):

    list_operations = ['Hit Rate (Train)','Hit Rate (Valid)', 'Hit Rate (Test)']
    action,actions = get_actions(treeview,position,list_operations)

    values = []
    names = []
    if action == actions['Hit Rate (Train)']:
        for item in epoch_items:
            hit = item.data_file[item.name+'/train/hit'].value
            names.append(item.name.split('/')[-1])
            values.append(rankingMetrics.hitrate(hit))

    if action == actions['Hit Rate (Valid)']:
        for item in epoch_items:
            hit = item.data_file[item.name+'/valid/hit'].value
            names.append(item.name.split('/')[-1])
            values.append(rankingMetrics.hitrate(hit))

    if action == actions['Hit Rate (Test)']:
        for item in epoch_items:
            if 'test' in item.data_file[item.name]:
                hit = item.data_file[item.name+'/test/hit'].value
                names.append(item.name.split('/')[-1])
                values.append(rankingMetrics.hitrate(hit))

    if haddock_item is not None:
        for item in haddock_item:
            hit = item.data_file[item.name+'/hitrate'].value
            names.append('haddock')
            values.append(hit)

    data_dict = {'_values':values,'_names':names}
    treeview.emitDict.emit(data_dict)

    data_dict = {}
    cmd  = "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
    cmd += "fig,ax = plt.subplots()\n"
    cmd += "for v,n in zip(_values,_names):"
    cmd += "    plt.plot(v,label=n)\n"
    cmd += "legen = ax.legend(loc='lower right')\n"
    cmd += "ax.set_xlabel('Top M')\n"
    cmd += "ax.set_ylabel('Hitrate')\n"
    cmd += "plt.show()\n"
    data_dict['exec_cmd'] = cmd
    treeview.emitDict.emit(data_dict)

def _context_multiple_epoch_multilevel(epoch_items,treeview,position,haddock_item=None):

    func_operations = {'Hit Rate':rankingMetrics.hitrate, 'Av. Prec.': rankingMetrics.avprec }
    list_operations = ['Hit Rate','Av. Prec.']
    list_subop = [['Train','Valid','Test'],['Train','Valid','Test']]
    action,actions = get_multilevel_actions(treeview,position,list_operations,list_subop)

    for iop,op in enumerate(list_operations):
        for subop in list_subop[iop]:
            if action == actions[(op,subop)]:
                plot_type,data_type = op,subop.lower()

    names, values =[], []
    for item in epoch_items:
        hit = item.data_file[item.name+'/'+data_type+'/hit'].value
        names.append(item.name.split('/')[-1])
        values.append(func_operations[plot_type](hit))

    data_dict = {'_values':values,'_names':names}
    treeview.emitDict.emit(data_dict)

    data_dict = {}
    cmd  = "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
    cmd += "fig,ax = plt.subplots()\n"
    cmd += "for v,n in zip(_values,_names):"
    cmd += "    plt.plot(v,label=n)\n"
    cmd += "legen = ax.legend(loc='lower right')\n"
    cmd += "ax.set_xlabel('Top M')\n"
    cmd += "ax.set_ylabel('%s')\n" %plot_type
    cmd += "plt.show()\n"
    data_dict['exec_cmd'] = cmd
    treeview.emitDict.emit(data_dict)


def _context_losses(item,treeview,position):

    menu = QtWidgets.QMenu()
    actions = {}
    list_operations = ['Plot Losses']

    for operation in list_operations:
        actions[operation] = menu.addAction(operation)
    action = menu.exec_(treeview.viewport().mapToGlobal(position))

    if action == actions['Plot Losses']:

        values = []
        train = item.data_file[item.name+'/train'].value
        valid = item.data_file[item.name+'/valid'].value
        values.append([x for x in train])
        values.append([x for x in valid])

        if 'test' in item.data_file[item.name]:
            test = item.data_file[item.name+'/test'].value
            values.append([x for x in test])

        data_dict = {'_values':values}
        treeview.emitDict.emit(data_dict)

        data_dict = {}
        cmd  = "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
        cmd += "fig,ax = plt.subplots()\n"
        cmd += "plt.plot(_values[0],c='red',label='train')\n"
        cmd += "plt.plot(_values[1],c='blue',label='valid')\n"

        if 'test' in item.data_file[item.name]:
            cmd += "plt.plot(_values[2],c='green',label='test')\n"

        cmd += "legen = ax.legend(loc='upper right')\n"
        cmd += "ax.set_xlabel('Epoch')\n"
        cmd += "ax.set_ylabel('Losses')\n"
        cmd += "plt.show()\n"
        data_dict['exec_cmd'] = cmd
        treeview.emitDict.emit(data_dict)
