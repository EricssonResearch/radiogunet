import torch
import os
from escnn import gspaces
from escnn import nn
from torchvision.transforms.functional import center_crop


def skip_connection(enc, dec):
    """ Combines encoder and decoder by cutting from the middle of
        the larger of the two and concatenate then
    """
    diff = enc.shape[-1]-dec.shape[-1]
    # Crop center
    if diff > 0:
        enc = center_crop(enc, output_size=dec.shape[2:])
    elif diff < 0:
        dec = center_crop(dec, output_size=enc.shape[2:])
    return nn.tensor_directsum([enc,dec])

def select_action_group(group_name):
    """ Converts a group name in string format to a valid
        action on the space.
        examples
        "C4" -> Equivariant under rotations by 90 degrees
        "C8" -> Equivariant under rotations by 45 degrees
        "D4" -> Equivariant under rotations by 90 degrees + reflections
        "D8" -> Equivariant under rotations by 45 degrees + reflections
        "S2" -> Equivariant under rotations by any degrees
    """        
    # Get number of rotations from name
    # numbers = ''.join(i for i in group_name if i.isdigit())
    # N = int(numbers)
    
    # if ("C" in group_name) and ("D" not in group_name):
    #     r2_act = gspaces.rot2dOnR2(N=N)
    # elif ("D" in group_name) and ("C" not in group_name):
    #     r2_act = gspaces.flipRot2dOnR2(N=N)
    if group_name == 'S2':
        r2_act = gspaces.rot2dOnR2(N=-1)
    else:
        raise ValueError("Incorrect group name")
    return r2_act


class S_UNet(torch.nn.Module):
    def __init__(self, config):
        super(S_UNet, self).__init__()
    
        # Mandatory input
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.channels = config['channels']
        
        # Optional input
        self.n_conv = config['n_conv'] if 'n_conv' in config else 2
        self.kernel_size = config['kernel_size'] if 'kernel_size' in config else 3
        self.padding = config['padding'] if 'padding' in config else 1
        self.activation = config['activation'] if 'activation' in config else 'relu'
        self.dropout = config['dropout'] if 'dropout' in config else 0
        self.batch_norm = config['batch_norm'] if 'batch_norm' in config else False
        self.base_num = config['base_num'] if 'base_num' in config else 8
        self.nl_N = config['point_num'] if 'point_num' in config else 16
        
        
        # Set the model equivariance under rotations/reflections
        # by specified group
        self.r2_act = select_action_group(config['group'])
        self.G = self.r2_act.fibergroup
        self.bl_regular_representation = self.G.bl_regular_representation(self.base_num)
        self.bl_irreps = self.G.bl_irreps(self.base_num)
        
        # Set input/output type as trivial representation
        self.input_type = nn.FieldType(self.r2_act, 
                                       self.in_channels*[self.r2_act.trivial_repr])
        self.output_type = nn.FieldType(self.r2_act, 
                                        self.out_channels*[self.r2_act.trivial_repr])        

        # Removes from an input image or feature map all the part of the signal 
        # defined on the pixels which lay outside the circle inscribed in the grid
        if config['equvariant_mask']:
            self.equivariant_mask = True
            self.MaskModule = nn.MaskModule(self.input_type, config['image_size'], margin=1)   
        else:
            self.equivariant_mask = False
            
        # Create list with encoder  channels
        enc_dims = [self.in_channels, *self.channels]
        
        # Convert to rt_act's
        enc_types, _, enc_nonlin1, _  = self.convert_to_r2_act(enc_dims)
        
        enc_types[0] = self.input_type # Manually convert to trivial_repr because input
        
        self.enc_conv = torch.nn.ModuleList([])
        self.enc_pool = torch.nn.ModuleList([])
        
        # Create encoder
        for i in range(1, len(enc_dims)):
            module_list = []
            module_list += self.get_layer(enc_types[i-1],enc_types[i], enc_nonlin1[i],
                                              self.batch_norm,self.dropout,self.activation)
            for _ in range(self.n_conv-1):
                module_list += self.get_layer(enc_types[i],enc_types[i], enc_nonlin1[i],
                                              self.batch_norm,self.dropout,self.activation)
            self.enc_conv.append(nn.SequentialModule(*module_list))
            
            # Create pooling seperatly
            self.enc_pool.append(nn.SequentialModule(
                nn.PointwiseAvgPoolAntialiased(enc_types[i],sigma=0.4, stride=2))
                )            
            
        # Create bottleneck
        nonlinearity = nn.FourierELU(self.r2_act, 2*enc_dims[-1], irreps=self.bl_irreps, N=self.nl_N, inplace=True)
        mid_type = nonlinearity.in_type
        bottleneck = []
        bottleneck += self.get_layer(enc_types[-1],mid_type,nonlinearity, self.batch_norm,self.dropout,self.activation)
        for _ in range(self.n_conv-1):
            bottleneck += self.get_layer(mid_type,mid_type,nonlinearity, self.batch_norm,self.dropout,self.activation)
        nonlinearity = nn.FourierELU(self.r2_act, enc_dims[-1], irreps=self.bl_irreps, N=self.nl_N, inplace=True)
        bottleneck += self.get_layer(mid_type,enc_types[-1],nonlinearity, self.batch_norm,self.dropout,self.activation)
        self.bottleneck = nn.SequentialModule(*bottleneck)

        # Create list with decoder channels
        dec_dims = [self.out_channels, *self.channels]
        dec_dims.reverse()
        
        # Convert to rt_act's
        dec_types,dec_types2, dec_nonlin1, dec_nonlin2  = self.convert_to_r2_act(dec_dims)
                
        self.dec_conv = torch.nn.ModuleList([])
        
        # Create decoder (upsampling)
        for i in range(1, len(dec_dims)):
            module_list = []
            module_list += self.get_layer(dec_types2[i-1],dec_types[i-1], dec_nonlin1[i-1],
                                              self.batch_norm,self.dropout,self.activation)
            
            module_list += self.get_layer(dec_types[i-1],dec_types[i], dec_nonlin1[i],
                                              self.batch_norm,self.dropout,self.activation)
            for _ in range(self.n_conv-2):
                # Add multiple convolutions in a level
                module_list += self.get_layer(dec_types[i],dec_types[i], dec_nonlin1[i],
                                              self.batch_norm,self.dropout,self.activation)

            self.dec_conv.append(nn.SequentialModule(*module_list))
           
        # Create upsampling
        self.dec_upsample = torch.nn.ModuleList([])
        for i in range(len(dec_dims)-1):
            self.dec_upsample.append(
                nn.SequentialModule(
                nn.R2ConvTransposed(dec_types[i],dec_types[i],kernel_size=4,
                                              stride=2,padding=1))
            )
        # Create tail end
        self.tail_conv = nn.SequentialModule(*self.get_layer(dec_types[-1],self.output_type,dec_nonlin1[i],False,0,None))

        return None

    def get_layer(self, in_type, out_type, nonlinearity,
                  bn = False, dropout = 0, activation = 'relu'):
        """
        Creates a convolutional layer between two dimensions acording to 
        specifications in the class.
        """
        out_list = []
        # Add convolutional layer
        out_list.append(
            nn.R2Conv(in_type, out_type, kernel_size=self.kernel_size,
                      padding=self.padding, bias=False)
        )
    
        # add batch norm
        if bn:
            out_list.append(
                nn.IIDBatchNorm2d(out_type)
            )
    
        # add activation
        if activation is None:
            pass
       
        elif activation == 'fourierELU':
            G = self.r2_act.fibergroup
            size = len(out_type.representations)
            # out_list.append(nn.FourierELU(self.r2_act, size, irreps=self.bl_irreps, N=self.nl_N, inplace=True))
            out_list.append(nonlinearity)
        else:
            raise Exception('Specified activation function not implemented!')
    
        # add dropout
        if dropout != 0:
            out_list.append(nn.PointwiseDropout(out_type,p=dropout))        
        return out_list
    
    def convert_to_r2_act(self,dims):
        """ dims is a list of dimensions that should be converted with 
            equivariance rotation acts
            e.g. dims = [1,4,8,16]
        """
        types,types_x2 = [],[]
        nonlin1, nonlin2 = [],[]
        G = self.r2_act.fibergroup
        for i,dim in enumerate(dims):
            nonlinearity = nn.FourierELU(self.r2_act, dim, irreps=self.bl_irreps, N=self.nl_N, inplace=True)
            types.append(nonlinearity.in_type)
            nonlinearity_x2 = nn.FourierELU(self.r2_act, dim*2, irreps=self.bl_irreps, N=self.nl_N, inplace=True)
            types_x2.append(nonlinearity_x2.in_type)
            nonlin1.append(nonlinearity)
            nonlin2.append(nonlinearity_x2)
            # types.append(nn.FieldType(self.r2_act, 
            #                                 dim*[self.bl_regular_representation]))
            # types_x2.append(nn.FieldType(self.r2_act, 
            #                                 2*dim*[self.bl_regular_representation]))
        return types, types_x2, nonlin1, nonlin2

    def forward(self, x):
        # Convert to geometric tensor
        x = nn.GeometricTensor(x, self.input_type)
        # x = self.input_type(x)
        
        # Removes from an input image or feature map all the part of the signal 
        # defined on the pixels which lay outside the circle inscribed in the grid
        if self.equivariant_mask:
            x = self.MaskModule(x)
        enc = x # x is input of first encoder
        
        # Pass through the encoder
        enc_out = []
        for i in range(len(self.enc_conv)):
            # Pass through a single encoder layer
            enc = self.enc_conv[i](enc)

            # save the encoder output such that it can be used for skip connections
            enc_out.append(enc)

            # Downsample with convolutional pooling
            enc = self.enc_pool[i](enc)

        # Pass through the bottleneck
        b = self.bottleneck(enc)

        # Pass through the decoder
        dec = b
        enc_out.reverse()   # reverse such that it fits pass through decoder
        for i in range(len(self.dec_conv)):
            # Get input for decoder
            dec = self.dec_upsample[i](dec)
            dec = skip_connection(enc_out[i], dec)

            # Pass through a single decoder layer
            # print (self.dec_conv[i])
            dec = self.dec_conv[i](dec)
            
        # Perform tail convolutions
        x = self.tail_conv(dec) # no activation
        x = x.tensor
        return x

class S_RadioWNet (torch.nn.Module):
    def __init__ (self, config_first, config_second, model_path = './data/'):
        super (S_RadioWNet, self).__init__()
        self.first_GUNet = S_UNet(config_first)
        self.second_GUNet = S_UNet (config_second)
        self.model_path = model_path
            
    def load_first_GUNet (self):
        print ('loading weights of first trained S-UNet')
        self.first_GUNet.load_state_dict(torch.load(self.model_path+'best_first_model.mdl', weights_only=True))
    
    def load_second_GUNet (self):
        print ('loading weights of second trained S-UNet')
        self.second_GUNet.load_state_dict(torch.load(self.model_path+'best_second_model.mdl', weights_only=True))
    
    def forward (self, x, stage='first'):
        if stage == 'first':
            x = self.first_GUNet(x)
        elif stage == 'second':
            with torch.no_grad():
                y = self.first_GUNet(x)
            x = torch.concatenate([x,y],axis=1)
            x = self.second_GUNet(x)
        return x

if __name__=='__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    config = {
        'in_channels' : 3,              # Number of input channels
        'out_channels' : 1,             # Number of output channels
        'channels' : [6,10,12],           # Convolution channels in layers
        'image_size': 128,              # Size of the images
        'n_conv' : 2,                   # Number of convolutions >= 2
        'batch_norm' : False,           # Use batch-norm (True/False)
        'dropout' : 0,                  # Use dropout? (0 <= dropout < 1)
        'kernel_size' : 3,              # Kernel size of all convolutions
        'padding' : 1,                  # Padding on all convolutions
        'equvariant_mask' : True,      # Whether to mask for equivariance
        'group'   : "S2",                # Which rotation group network belongs to
        'activation': "norm"
        }
    
    model = S_UNet(config).to(device)
    x = torch.rand((1,3,128,128)).to(device)
    y = model(x)
    print(y.shape)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:",pytorch_total_params)

    
    
    
    
    
    
    

    
    