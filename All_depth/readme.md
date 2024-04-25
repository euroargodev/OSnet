OSnet "all depth"
This is an updated version of OSnet, that allows hydrographic profiles to be of different length. This is still in developpement. The preprocessing includes now a threshold selection of the shortest profiles to take into account in the analysis. 

Workflow :

- Preprocessing 1 to 5

- NN_train : train an network with all the observed profiles.

- Pred_full : use the models produced in NN_train to predict the profiles observed.

- apply_MLD_mask_all_depth : find the best value for MLD and apply the MLD adjustment on the predicted dataset. 

- product_generation_all_depth.py : make a gridded product from the surface, down to 4000m.

- ADjust_product.py : adjust MLD for the product.
