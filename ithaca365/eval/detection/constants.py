# Ithaca365 dev-kit.

DETECTION_NAMES = ['car', 'truck', 'bus',
                   'pedestrian', 'motorcyclist', 'bicyclist']

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'pedestrian': 'Pedestrian',
                          'motorcyclist': 'Motorcyclist',
                          'bicyclist': 'Bicyclist',}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'pedestrian': 'C3',
                    'motorcyclist': 'C4',
                    'bicyclist': 'C5',
                    }

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'object.valid']
# ATTRIBUTE_NAMES = ['object.valid']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped',
                          'object.valid': 'Valid Object'}
# PRETTY_ATTRIBUTE_NAMES = {'object.valid': 'Valid Object'}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.'}
