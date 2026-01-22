package fr.ensimag.deep.serialization.json;

import fr.ensimag.deep.trainer.initializer.Gaussian;
import fr.ensimag.deep.trainer.initializer.He;
import fr.ensimag.deep.trainer.initializer.INewtorkInitializer;
import fr.ensimag.deep.trainer.initializer.Xavier;

public class InitializerFactory {
    public static INewtorkInitializer create(String initializer)
    {
        INewtorkInitializer result = null;

        if ("Gaussian".equals(initializer))
            result = new Gaussian();
        else if ("Xavier".equals(initializer))
            result = new Xavier();   
        else if ("He".equals(initializer))
            result = new He();

        return result;
    }
}
