# Original project brief

The notes this repository started from. Kept so the requirements stay auditable against what was
built — see the main [README](../README.md) for the result.

---

# py123detection
py123d toolkit for detection tasks

## MMCV Plugin
- I want to write a script that can translate any dataset that is in 123D format into a data object that can be used to train models in mmcv ecosystem

- Note: its likely that for conversion and training you need to python environment which should be okay (as mmcv usually requires 3.8): is that an issue when saving pickle and reading pickle under two different python versions?

- Check under the folder PETR/ and StreamPETR/ to see what the requirements are. Also look at CoIn3D as they do mixed training, so if we want 123D to be part of the ecosystem it should support that as well

- Consider the case that for 123d sometimes datasets are not stored with an explicit copy but rather references to the images, consider that (i.e. 123D allows to store reference for image or stores image as binary)

- Write a conversion script for 123D into a pickle (or whatever format) that is required by mmcv ecosystem

- Find a good way to read the scenes API and convert it (also allow sample rate etc). In particular cross dataset experiments as done in CoIn3D require them, also take a look how it was done in master-thesis

- But find a solution that is the easiest to work with. Think whether loading scene APIs or if it makes sense to just pass 123D dataset folder name or so

- Do test cases, convert nuscenes mini to 123d and then to mmcv format and compare when directly transfering to mmcv format

## Py123Detection
- My goal is to have py123detection as tool that uses 123D for 3d multi view object detection
- It should have two branches: one where we convert the data to mmcv format and people can continue like that
- And second branch would be more focused on master-thesis (in which we ditched mmcv format and work with more modern stack); but for now focus on first branch

I provided you with a nuscenes sample kit "v1.0-mini". use it to convert it to 123d, from there to pickle and compare when direclty ocnveritng to pickle
