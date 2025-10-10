from typing import List, Union, Dict, Tuple, Optional
import importlib, types


def module_exists(
    *names: Union[List[str], str],
    error: str = "ignore",
    warn_every_time: bool = False,
    __INSTALLED_OPTIONAL_MODULES: Dict[str, bool] = {},
) -> Optional[Union[Tuple[types.ModuleType, ...], types.ModuleType]]:
    """
    Try to import optional dependencies.
    Ref: https://stackoverflow.com/a/73838546/4900327

    Parameters
    ----------
    names: str or list of strings.
        The module name(s) to import.
    error: str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found.
        * raise : Raise an ImportError.
        * warn: print a warning.
        * ignore: If any module is not installed, return None, otherwise,
          return the module(s).
    warn_every_time: bool
        Whether to warn every time an import is tried. Only applies when error="warn".
        Setting this to True will result in multiple warnings if you try to
        import the same library multiple times.
    Returns
    -------
    maybe_module : Optional[ModuleType, Tuple[ModuleType...]]
        The imported module(s), if all are found.
        None is returned if any module is not found and `error!="raise"`.
    """
    assert error in {"raise", "warn", "ignore"}
    if isinstance(names, (list, tuple, set)):
        names: List[str] = list(names)
    else:
        assert isinstance(names, str)
        names: List[str] = [names]
    modules = []
    for name in names:
        try:
            module = importlib.import_module(name)
            modules.append(module)
            __INSTALLED_OPTIONAL_MODULES[name] = True
        except ImportError:
            modules.append(None)

    def error_msg(missing: Union[str, List[str]]):
        if not isinstance(missing, (list, tuple)):
            missing = [missing]
        missing_str: str = " ".join([f'"{name}"' for name in missing])
        dep_str = "dependencies"
        if len(missing) == 1:
            dep_str = "dependency"
        msg = f"Missing optional {dep_str} {missing_str}. Use pip or conda to install."
        return msg

    missing_modules: List[str] = [name for name, module in zip(names, modules) if module is None]
    if len(missing_modules) > 0:
        if error == "raise":
            raise ImportError(error_msg(missing_modules))
        if error == "warn":
            for name in missing_modules:
                ## Ensures warning is printed only once
                if warn_every_time is True or name not in __INSTALLED_OPTIONAL_MODULES:
                    print(f"Warning: {error_msg(name)}")
                    __INSTALLED_OPTIONAL_MODULES[name] = False
        return None
    if len(modules) == 1:
        return modules[0]
    return tuple(modules)
