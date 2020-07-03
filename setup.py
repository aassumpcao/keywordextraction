from setuptools import setup, find_packages

setup(
    name='keywordextraction',
    version='0.1',
    url='https://github.com/aassumpcao/keywordextraction',
    author='Andre Assumpcao',
    author_email='andre.assumpcao@gmail.com',
    packages=find_packages(),
    description='This package performs keyword extraction from texts.',
    license='MIT',
    # scripts='keywordextraction.py',
    install_requires=[
        'contractions', 'numpy', 'spacy', 'scipy', 'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
