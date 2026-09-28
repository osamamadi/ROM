dataset_info = dict(
    dataset_name='ankle_v2_malleolus',
    paper_info=dict(author='Braude Phase B', title='Markerless ankle ROM',
                    container='', year='2026', homepage=''),
    keypoint_info={
        0: dict(name='bottom_heel',    id=0, color=[230, 25, 75],
                type='lower', swap=''),
        1: dict(name='5th_metatarsal', id=1, color=[60, 180, 75],
                type='lower', swap=''),
        2: dict(name='malleolus',      id=2, color=[67, 99, 216],
                type='lower', swap=''),
    },
    skeleton_info={
        0: dict(link=('bottom_heel', '5th_metatarsal'), id=0, color=[0, 255, 255]),
        1: dict(link=('bottom_heel', 'malleolus'),      id=1, color=[255, 0, 255]),
    },
    joint_weights=[1.5, 1.5, 1.0],
    sigmas=[0.025, 0.025, 0.035],
)
