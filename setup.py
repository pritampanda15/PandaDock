from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f 
                    if line.strip() and not line.startswith('#')]

# Optional dependencies for different use cases
extras_require = {
    'gpu': [
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'cupy-cuda11x>=11.0.0'
    ],
    'ml': [
        'scikit-optimize>=0.9.0',
        'GPy>=1.10.0'
    ],
    'vis': [
        'seaborn>=0.11.0',
        'plotly>=5.7.0'
    ],
    'full': [
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'cupy-cuda11x>=11.0.0',
        'scikit-optimize>=0.9.0',
        'GPy>=1.10.0',
        'seaborn>=0.11.0',
        'plotly>=5.7.0',
        'jupyter>=1.0.0'
    ]
}

setup(
    name='pandadock',
    version='1.0.0',
    description='Physics-Based Molecular Docking Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Pritam Kumar Panda',
    author_email='pritam@stanford.edu',
    url='https://github.com/pritampanda15/PandaDock',
    packages=find_packages(exclude=['tests*', 'docs*']),
    install_requires=requirements,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    keywords='molecular-docking computational-chemistry bioinformatics',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'pandadock=pandadock.main:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/pritampanda15/PandaDock/issues',
        'Source': 'https://github.com/pritampanda15/PandaDock',
        'Documentation': 'https://github.com/pritampanda15/PandaDock/blob/main/README.md',
    },
    include_package_data=True,  # Include files specified in MANIFEST.in
)
