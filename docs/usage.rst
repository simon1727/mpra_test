=====
Usage
=====

To use MPRA Test in a project:

```python
import mpra_test
```

To load the plant enhancer ACR dataset:

```python
mpra_dataset = mpra_test.MPRA_Dataset.load('Plant_2024_Jores', 'native')
```

To acccess the dataset object:

```python
print(mpra_dataset[:3].shape)
print(mpra_dataset[[0,1,2]].shape)
print(mpra_dataset[np.arange(3)].shape)
print(mpra_dataset[torch.arange(3)].shape)

print(mpra_dataset.Y['dark'].shape)
print(mpra_dataset.Y[['cold', 'warm', 'dark', 'light']].shape)

print(mpra_dataset[mpra_dataset.Y['dark'] > 3].shape)
print(mpra_dataset[mpra_dataset.obs_X['chr'].isin([2, 3, 5, 7])].shape)
```

or:

```python
print(mpra_dataset['Y: dark'].shape)
print(mpra_dataset[['Y: cold', 'Y: warm', 'Y: dark', 'Y: light']].shape)

print(mpra_dataset[mpra_dataset['Y: dark'] > 3].shape)
print(mpra_dataset[mpra_dataset['obs_X: chr'].isin([2, 3, 5, 7])].shape)
```
