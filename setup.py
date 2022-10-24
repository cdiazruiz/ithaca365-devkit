from setuptools import setup, find_packages

setup(name='ithaca365',
      version='1.0',
      description='The official devkit of the Ithaca-365 dataset (https://ithaca365.mae.cornell.edu/).',
      author='Carlos Andres Diaz, Youya Xia, Yurong You, Jose Nino, Junan Chen, Josephine Monica, Xiangyu Chen, Katie Z Luo, Yan Wang, Marc Emond, Wei-Lun Chao, Bharath Hariharan, Kilian Q. Weinberger, Mark Campbell',
      author_email='cad297@cornell.edu',
      url='https://ithaca365.mae.cornell.edu/',
      python_requires='>=3.6',
      packages=find_packages(),
      install_requires=['opencv-python', 'numpy', 'Pillow', 'matplotlib', 'importlib_metadata', 'scikit-learn', 'pyquaternion', 'tqdm', 'cachetools'])