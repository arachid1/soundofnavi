# $1 is the vm name
# if running with v100 replace image to the tensorflow enabled image, location us-east-2 and 

az vm create \
    --resource-group training \
    --name $1 \
    --image Canonical:UbuntuServer:20.04-LTS:latest \
    --assign-identity /subscriptions/6a181d75-cf16-43f7-8e7d-4cf2ca9d743c/resourcegroups/training/providers/microsoft.managedidentity/userassignedidentities/vm_identity \
    --custom-data cloudconfig.yaml \
    --data-disk-sizes-gb 1024 --size Standard_DS2_v2 \
    --boot-diagnostics-storage \
    --location us-east
    --generate-ssh-keys