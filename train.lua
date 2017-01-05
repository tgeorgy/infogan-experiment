require 'dataset-mnist'
require 'nngraph'
require 'optim'
require 'paths'

opt = {
    gpu = 1,              -- GPU ID, set to 0 to run on CPU
    noise_param_n = 64,   -- Number of noise paramteres
    cont_param_n = 2,     -- Continues latent parameters number
    desc_param_n = 10,    -- Descrete latent parameters number
    batch_size = 64,      -- Batch size
    img_chan_n = 1,       -- Number of image channels (1 for gray scaled)
    seed = 1,             -- Random generator seed
    info_coef = 1,        -- Infogan lambda coefficient (info loss weight)
}

-- parse environment parameters to override defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or v end
print(opt)

opt.image_size = 32  -- Image height and width. Fixed for now.

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- load data
mnist.download()
local train_set = mnist.loadTrainSet()

----------------------------------------------------------------------------
local total_parameters_n = opt.noise_param_n + opt.cont_param_n + opt.desc_param_n

-- Generator initialization
local netG = nn.Sequential()

netG:add(nn.Linear(total_parameters_n, 1024))
netG:add(nn.BatchNormalization(1024))
netG:add(nn.ReLU(true))

netG:add(nn.Linear(1024, 128 * 8 * 8))
netG:add(nn.BatchNormalization(128 * 8 * 8))
netG:add(nn.ReLU(true))
netG:add(nn.Reshape(128, 8, 8))

netG:add(nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(64))
netG:add(nn.ReLU(true))

netG:add(nn.SpatialFullConvolution(64, 1, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())

-- Discriminator initialization
local netD = nn.Sequential()

local netD_input = nn.Identity()()
netD:add(nn.SpatialConvolution(1, 64, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))

netD:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(128))
netD:add(nn.LeakyReLU(0.2, true))

netD:add(nn.Reshape(128 * 8 * 8))
netD:add(nn.Linear(128 * 8 * 8, 1024))
netD:add(nn.BatchNormalization(1024))
netD:add(nn.LeakyReLU(0.2, true))

netD = netD(netD_input)

local netQ = nn.Linear(1024, 128)(netD)
netQ = nn.BatchNormalization(128)(netQ)
netQ = nn.LeakyReLU(0.2, true)(netQ)
netQ = nn.Linear(128, opt.cont_param_n + opt.desc_param_n)(netQ)

local netQ_c = nn.Narrow(2, 1, opt.cont_param_n)(netQ)

local netQ_d = nn.Narrow(2, opt.cont_param_n + 1, opt.desc_param_n)(netQ)
netQ_d = nn.LogSoftMax()(netQ_d)

netD = nn.Linear(1024, 1)(netD)
netD = nn.Sigmoid()(netD)

netD = nn.gModule({netD_input}, {netD, netQ_c, netQ_d})

local criterion_D = nn.BCECriterion()
local criterion_Q_d = nn.ClassNLLCriterion()
local criterion_Q_c = nn.MSECriterion()
----------------------------------------------------------------------------

local input = torch.Tensor(opt.batch_size, 1, opt.image_size, opt.image_size)
local noise = torch.Tensor(opt.batch_size, total_parameters_n)
local label = torch.Tensor(opt.batch_size)
local latent_label = torch.LongTensor(opt.batch_size)

if opt.gpu > 0 then
    require 'cunn'

    cutorch.setDevice(opt.gpu)
 
    if pcall(require, 'cudnn') then
        require 'cudnn'
        cudnn.benchmark = true
        cudnn.convert(netG, cudnn)
        cudnn.convert(netD, cudnn)
    end

    criterion_D:cuda()
    criterion_Q_d:cuda()
    criterion_Q_c:cuda()
    netD:cuda()
    netG:cuda()

    input = input:cuda()
    noise = noise:cuda()
    label = label:cuda()
    latent_label = latent_label:cuda()
end
----------------------------------------------------------------------------
function provide_data(batch_size)
    local latent_param_c = torch.rand(opt.batch_size, opt.cont_param_n) * 2 - 1
    local latent_id = torch.multinomial(
        torch.ones(opt.desc_param_n),
        opt.batch_size,
        true)

    local latent_param_d = torch.zeros(opt.batch_size, opt.desc_param_n)
    latent_param_d:scatter(2, latent_id:reshape(opt.batch_size, 1), torch.ones(opt.batch_size, 1))
    local latent_noise = torch.rand(opt.batch_size, opt.noise_param_n) * 2 - 1

    local latent_param = torch.cat({latent_param_c, latent_param_d, latent_noise}, 2)
    local img_ids = torch.LongTensor(opt.batch_size):random(1, train_set.size())
    local real_img = train_set.data:index(1, img_ids)
    return latent_param, latent_id, real_img:float()*2/255-1
end
----------------------------------------------------------------------------
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    gradParametersD:zero()

    latent_param, latent_id, real_img = provide_data(opt.batch_size)
    -- train with real
    input:copy(real_img)
    noise:copy(latent_param)
    label:fill(1)

    local output = netD:forward(input)
    local output_D = output[1]
    local output_Q_c = output[2]
    local output_Q_d = output[3]

    local errD_real = criterion_D:forward(output_D, label)

    local df_do_D = criterion_D:backward(output_D, label)
    local df_do_Q_c = output_Q_c:clone():zero()
    local df_do_Q_d = output_Q_d:clone():zero()
    netD:backward(input, {df_do_D, df_do_Q_c, df_do_Q_d})

    -- train with fake
    local fake = netG:forward(noise)
    input:copy(fake)
    label:fill(0)

    output = netD:forward(input)
    local output_D = output[1]
    local output_Q_c = output[2]
    local output_Q_d = output[3]

    local errD_fake = criterion_D:forward(output_D, label)
    local label_Q_c = noise:narrow(2, 1, opt.cont_param_n)
    latent_label:copy(latent_id)
    errQ_c = criterion_Q_c:forward(output_Q_c, label_Q_c)
    errQ_d = criterion_Q_d:forward(output_Q_d, latent_label)

    local df_do_D = criterion_D:backward(output_D, label)
    local df_do_Q_c = criterion_Q_c:backward(output_Q_c, label_Q_c)
    local df_do_Q_d = criterion_Q_d:backward(output_Q_d, latent_label)
    netD:backward(input, {df_do_D, df_do_Q_c*opt.info_coef, df_do_Q_d*opt.info_coef})

    errD = (errD_real + errD_fake)/2
    errQ = (errQ_c + errQ_d)/2

    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    gradParametersG:zero()

    label:fill(1)

    local output = netD.output
    errG = criterion_D:forward(output[1], label)
    local df_do_D = criterion_D:backward(output[1], label)
    local df_do_Q_c = criterion_Q_c.gradInput
    local df_do_Q_d = criterion_Q_d.gradInput
    local df_dg = netD:updateGradInput({input}, {df_do_D, df_do_Q_c*opt.info_coef, df_do_Q_d*opt.info_coef})

    netG:backward(noise, df_dg)
    return errG, gradParametersG
end
----------------------------------------------------------------------------
local optimStateD = {
   learningRate = 0.0002,
   beta1 = 0.5,
}
local optimStateG = {
   learningRate = 0.001,
   beta1 = 0.5,
}

if paths.dir('checkpoint') == nil then
    paths.mkdir('checkpoint')
end

netG:training()
netD:training()

local nepochs = 20
local print_every = 50

for epochi = 1,nepochs do
    for batchi = 1,train_set.size()/opt.batch_size do
        optim.adam(fDx, parametersD, optimStateD)
        optim.adam(fGx, parametersG, optimStateG)

        if batchi%print_every == 0 then
            print(('Epoch:%d \tBatch:%d\nErr_G:%0.4f\tErr_D:%0.4f'):format(epochi, batchi, errG, errD))
        end
    end
    torch.save('checkpoint/netG.t7', netG:clearState())
    torch.save('checkpoint/netD.t7', netD:clearState())
end
