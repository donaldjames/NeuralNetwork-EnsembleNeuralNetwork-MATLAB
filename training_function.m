function [performance, percentErrors, y] = training_function(net, x, t)
    [net, tr] = train(net,x,t);
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
end